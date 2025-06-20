import cv2
import numpy as np
import time
from ultralytics import YOLO
from scipy.spatial import distance
import datetime
import os
import collections
import configparser
import ast
import threading 
import ctypes 

# --- VLC Imports for RTSP Streaming ---
import vlc

# --- Google Sheets API Imports ---
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- PyDrive Imports for Google Drive Upload ---
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# --- Configuration Reading Function ---
def read_config(config_file='YOUR CONFIG FILE PATH'):
    """Reads configuration parameters from the specified INI file."""
    config = configparser.ConfigParser()
    
    if not os.path.exists(config_file):
        print(f"Error: Configuration file '{config_file}' not found.")
        print("Please create it based on the provided template.")
        exit()

    config.read(config_file)
    
    params = {}

    try:
        # GOOGLE_SHEETS
        params['google_sheet_id'] = config.get('GOOGLE_SHEETS', 'google_sheet_id')

        # GOOGLE_DRIVE
        params['gdrive_root_folder_id'] = config.get('GOOGLE_DRIVE', 'gdrive_root_folder_id')
        if params['gdrive_root_folder_id'].strip() == '':
            params['gdrive_root_folder_id'] = None

        # MODEL_CONFIG
        params['yolo_model_path'] = config.get('MODEL_CONFIG', 'yolo_model_path')

        # APP_PARAMETERS (Parse as floats/ints)
        params['conf_threshold'] = config.getfloat('APP_PARAMETERS', 'conf_threshold')
        params['max_distance_for_track'] = config.getint('APP_PARAMETERS', 'max_distance_for_track')
        params['max_frames_to_skip'] = config.getint('APP_PARAMETERS', 'max_frames_to_skip')
        params['stop_duration_seconds'] = config.getfloat('APP_PARAMETERS', 'stop_duration_seconds')
        params['violation_display_duration'] = config.getfloat('APP_PARAMETERS', 'violation_display_duration')
        params['static_movement_threshold_pixels'] = config.getint('APP_PARAMETERS', 'static_movement_threshold_pixels')
        # params['static_frames_needed'] = config.getint('APP_PARAMETERS', 'static_frames_needed') # No longer directly used for wait logic
        params['max_tolerated_frames_for_wait'] = config.getint('APP_PARAMETERS', 'max_tolerated_frames_for_wait') # New parameter
        params['direction_movement_threshold_pixels'] = config.getint('APP_PARAMETERS', 'direction_movement_threshold_pixels')
        params['buffer_duration_seconds'] = config.getfloat('APP_PARAMETERS', 'buffer_duration_seconds')
        params['clip_pre_event_buffer_seconds'] = config.getfloat('APP_PARAMETERS', 'clip_pre_event_buffer_seconds')
        params['clip_post_event_buffer_seconds'] = config.getfloat('APP_PARAMETERS', 'clip_post_event_buffer_seconds')
        params['yolo_inference_fps'] = config.getint('APP_PARAMETERS', 'yolo_inference_fps') # For live processing rate

        # CAMERA_CONFIGS
        camera_ids_str = config.get('CAMERA_CONFIGS', 'camera_ids')
        params['camera_ids'] = [cid.strip() for cid in camera_ids_str.split(',') if cid.strip()]
        
        if not params['camera_ids']:
            print("Error: No camera IDs found in 'camera_ids' under [CAMERA_CONFIGS].")
            exit()
        
        # We process only the first camera ID for simplicity in this example
        first_camera_id = params['camera_ids'][0]
        params['video_source'] = config.get(first_camera_id.upper(), 'video_path') # Renamed to video_source
        params['output_processed_video_path'] = config.get(first_camera_id.upper(), 'output_processed_video_path')
        params['location'] = config.get(first_camera_id.upper(), 'location')
        
        # LINES_COORDINATES specific to the selected camera
        params['blue_line_start'] = ast.literal_eval(config.get(first_camera_id.upper(), 'blue_line_start'))
        params['blue_line_end'] = ast.literal_eval(config.get(first_camera_id.upper(), 'blue_line_end'))
        params['green_line1_start'] = ast.literal_eval(config.get(first_camera_id.upper(), 'green_line1_start'))
        params['green_line2_start'] = ast.literal_eval(config.get(first_camera_id.upper(), 'green_line2_start'))

        # Add line colors and thickness to params
        params['blue_color'] = (255, 0, 0)
        params['line_thickness'] = 4
        params['green_color'] = (0, 255, 0)
        params['green_line_thickness'] = 4

        # PARKING_ZONE_CONFIG (Add if present in config.ini, else default)
        try:
            params['parking_zone_point1'] = ast.literal_eval(config.get('PARKING_ZONE_CONFIG', 'parking_zone_point1'))
            params['parking_zone_point2'] = ast.literal_eval(config.get('PARKING_ZONE_CONFIG', 'parking_zone_point2'))
            params['parking_zone_point3'] = ast.literal_eval(config.get('PARKING_ZONE_CONFIG', 'parking_zone_point3'))
            params['parking_zone_point4'] = ast.literal_eval(config.get('PARKING_ZONE_CONFIG', 'parking_zone_point4'))
        except (configparser.NoSectionError, configparser.NoOptionError):
            print("Warning: Parking zone coordinates not found in config.ini. Parking zone will not be drawn.")
            params['parking_zone_point1'] = None # Indicate no parking zone

        # Use the global temp_output_clips_dir and temp_violation_images_dir
        params['temp_output_clips_dir'] = config.get('PATHS', 'temp_output_clips_dir')
        params['temp_violation_images_dir'] = config.get('PATHS', 'temp_violation_images_dir')


    except (configparser.NoSectionError, configparser.NoOptionError, ValueError) as e:
        print(f"Error parsing config.ini: {e}")
        print("Please ensure all sections and options are correctly defined in config.ini and values are in the correct format.")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred while reading config.ini: {e}")
        exit()

    return params

# --- Read configuration at the very beginning ---
config_params = read_config()

# --- Configuration for YOLOv5 ---
model = YOLO(config_params['yolo_model_path'])
CONF_THRESHOLD = config_params['conf_threshold'] # This will be accessed via config_params within the class
vehicle_class_names = [    # Only detect these specific vehicle types
    'car', 'motorcycle', 'bus', 'truck', 'bicycle'
]

# --- Global VLC Frame Buffer for processing ---
pixels_initial_width = 1280
pixels_initial_height = 720
pixels_initial_pitch = pixels_initial_width * 4 # Assuming RV32 (4 bytes per pixel)
pixels = np.zeros((pixels_initial_height, pixels_initial_pitch), dtype=np.uint8)

vlc_video_dims = {'width': pixels_initial_width, 'height': pixels_initial_height, 'pitch': pixels_initial_pitch}
pixels_lock = threading.Lock()

# Define the C function types for VLC callbacks
LockCb = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))
UnlockCb = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))
DisplayCb = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)


# VLC Callback functions for memory output
@LockCb
def video_lock_cb(opaque, planes):
    """
    Called by VLC when it needs a buffer to write video data into.
    We provide a pointer to our numpy array.
    """
    global pixels, vlc_video_dims
    with pixels_lock:
        current_width = vlc_video_dims['width']
        current_height = vlc_video_dims['height']
        current_pitch = vlc_video_dims['pitch']

        # Ensure pixels array is large enough for current dimensions
        required_size = current_height * current_pitch
        if pixels.nbytes < required_size:
            pixels = np.zeros((current_height, current_pitch), dtype=np.uint8)
        
        planes[0] = ctypes.cast(pixels.ctypes.data, ctypes.c_void_p) 
    return None

@UnlockCb
def video_unlock_cb(opaque, picture, planes):
    """
    Called by VLC after it has written video data into the buffer.
    """
    pass

@DisplayCb
def video_display_cb(opaque, picture):
    """
    Called by VLC to signal that a new picture is ready for display.
    This callback is typically used for GUI display.
    In headless mode, it doesn't perform any display actions.
    """
    pass


# --- Google Sheets Logging Variables ---
gspread_scope = ["https://www.googleapis.com/auth/spreadsheets"]
credentials = None
gc = None
worksheet = None
log_serial_number = 0

# --- Google Sheets Cell Formatting Colors ---
RED_FORMAT = {
    "backgroundColor": {
        "red": 1.0,
        "green": 0.0,
        "blue": 0.0
    }
}

YELLOW_FORMAT = {
    "backgroundColor": {
        "red": 1.0,
        "green": 1.0,
        "blue": 0.0
    }
}

GREEN_FORMAT = {
    "backgroundColor": {
        "red": 0.0,
        "green": 1.0,
        "blue": 0.0
    }
}

# --- Global variables for Google Drive Upload ---
drive = None
gdrive_folder_cache = {}

# Create the temporary directory for clips if it's not exist
if not os.path.exists(config_params['temp_output_clips_dir']):
    os.makedirs(config_params['temp_output_clips_dir'])
    print(f"Created temporary directory for clips: {config_params['temp_output_clips_dir']}")
else:
    print(f"Temporary clips directory already exists: {config_params['temp_output_clips_dir']}")

# Create the temporary directory for images
if not os.path.exists(config_params['temp_violation_images_dir']):
    os.makedirs(config_params['temp_violation_images_dir'])
    print(f"Created temporary directory for images: {config_params['temp_violation_images_dir']}")
else:
    print(f"Temporary images directory already exists: {config_params['temp_violation_images_dir']}")

# --- Google Sheets Logging Functions ---
def initialize_google_sheet_logger(google_sheet_id): # Pass google_sheet_id
    global credentials, gc, worksheet, log_serial_number

    try:
        credentials = ServiceAccountCredentials.from_json_keyfile_name('service_account.json', gspread_scope)
        gc = gspread.authorize(credentials)
        
        spreadsheet = gc.open_by_key(google_sheet_id)
        worksheet = spreadsheet.get_worksheet(0)

        headers = ["Serial Number", "Time Stamp", "Location", "Vehicle Type", "Direction(In/Out)", "STOP Status", "Wait Status", "Violation", "Video Link", "Image Link"]
        
        existing_headers = worksheet.row_values(1)
        if not existing_headers or existing_headers != headers:
            if not existing_headers:
                worksheet.append_row(headers)
                print("Headers added to Google Sheet (including 'Location', 'Video Link', and 'Image Link' columns).")
            else:
                print("Warning: Existing Google Sheet headers do not match required headers.")
                print("Expected:", headers)
                print("Found:", existing_headers)
                print("Please ensure the first row of your Google Sheet matches the expected headers,")
                print("especially the new 'Location', 'Video Link', and 'Image Link' columns, or clear the sheet to re-add headers.")
        
        log_serial_number = len(worksheet.get_all_values()) - 1
        if log_serial_number < 0:
            log_serial_number = 0

        print(f"Google Sheet logger initialized for Sheet ID: {google_sheet_id}")
        print(f"Starting serial number for new entries: {log_serial_number + 1}")

    except Exception as e:
        print(f"Error initializing Google Sheet logger: {e}")
        print("Please ensure 'service_account.json' is in the same directory and the Google Sheet is shared with the service account email.")
        print(f"Also verify the 'google_sheet_id' in the script is correct.")
        return # Exit the function, not the whole script

def log_vehicle_data(timestamp, location, vehicle_type, direction, stop_status, wait_status, violation, video_link="", image_link=""):
    global worksheet, log_serial_number
    
    if worksheet is None:
        print("Google Sheet not initialized. Cannot log data.")
        return

    log_serial_number += 1
    row_data = [log_serial_number, timestamp, location, vehicle_type, direction, stop_status, wait_status, violation, video_link, image_link]
    
    try:
        worksheet.append_row(row_data)
        print(f"Logged vehicle data to Google Sheet (Serial: {log_serial_number}).")

        sheet_row_index = log_serial_number + 1

        format_properties = None
        if violation == "Yes":
            format_properties = RED_FORMAT
            print(f"Applied color 'red' to row {sheet_row_index} (Violation: Yes).")
        elif direction == "Out":
            format_properties = YELLOW_FORMAT
            print(f"Applied color 'yellow' to row {sheet_row_index} (Direction: Out).")
        elif violation == "No":
            format_properties = GREEN_FORMAT
            print(f"Applied color 'green' to row {sheet_row_index} (Violation: No, Direction: In).")
        
        if format_properties:
            worksheet.format(f"A{sheet_row_index}:J{sheet_row_index}", format_properties)

    except Exception as e:
        print(f"Error appending row or formatting to Google Sheet: {e}")
        print("Check internet connection, API quotas, or sheet permissions.")

# --- Google Drive Upload Functions ---

def initialize_google_drive(GDriveRootFolderId): # Pass GDriveRootFolderId
    global drive
    try:
        gauth = GoogleAuth()
        gauth.LoadCredentialsFile("mycreds.txt")
        if gauth.credentials is None:
            gauth.LocalWebserverAuth()
        elif gauth.access_token_expired:
            gauth.Refresh()
        else:
            gauth.Authorize()
        
        gauth.SaveCredentialsFile("mycreds.txt")
        print("Google Drive authentication successful.")
        drive = GoogleDrive(gauth)

        if GDriveRootFolderId:
            try:
                drive.ListFile({'q': f"'{GDriveRootFolderId}' in parents and trashed=false"}).GetList()
                print(f"Using Google Drive Root Folder ID: {GDriveRootFolderId}")
            except Exception as e:
                print(f"Error: Google Drive Root Folder ID '{GDriveRootFolderId}' is invalid or inaccessible: {e}")
                print("Please ensure you've created the folder and copied its ID correctly.")
                print("Set GDriveRootFolderId to None or empty string to upload to the root of My Drive.")
                return # Exit the function, not the whole script
        else:
            print("No specific Google Drive Root Folder ID provided. 'Violations' and 'Non_Violations_By_Type' folders will be created in the root of My Drive.")

    except Exception as e:
        print(f"Error initializing Google Drive: {e}")
        print("Please ensure 'client_secrets.json' is in the same directory and Google Drive API is enabled.")
        print("Also, check your internet connection during the first-time authentication.")
        return # Exit the function, not the whole script

def get_or_create_gdrive_folder(folder_name, parent_folder_id=None):
    """
    Gets the ID of an existing Google Drive folder or creates it if it doesn't exist.
    Caches results to minimize API calls.
    """
    global drive
    cache_key = f"{folder_name}_{parent_folder_id or 'root'}"
    if cache_key in gdrive_folder_cache:
        return gdrive_folder_cache[cache_key]

    q_query = f"title='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent_folder_id:
        q_query += f" and '{parent_folder_id}' in parents"
    
    file_list = drive.ListFile({'q': q_query}).GetList()

    if file_list:
        folder_id = file_list[0]['id']
        gdrive_folder_cache[cache_key] = folder_id
        return folder_id
    else:
        file_metadata = {
            'title': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        if parent_folder_id:
            file_metadata['parents'] = [{'id': parent_folder_id}]
        
        folder = drive.CreateFile(file_metadata)
        folder.Upload()
        folder_id = folder['id']
        print(f"Created new Google Drive folder '{folder_name}' (ID: {folder_id})")
        gdrive_folder_cache[cache_key] = folder_id
        return folder_id

def upload_video_to_drive(file_path, file_name, target_folder_id=None):
    """
    Uploads a video file to Google Drive.
    If target_folder_id is provided, uploads to that folder.
    """
    global drive
    if drive is None:
        print("Google Drive not initialized. Cannot upload video.")
        return None

    try:
        file_metadata = {
            'title': file_name,
            'mimeType': 'video/mp4'
        }
        if target_folder_id:
            file_metadata['parents'] = [{'id': target_folder_id}]

        gfile = drive.CreateFile(file_metadata)
        gfile.SetContentFile(file_path)
        gfile.Upload()

        permission = gfile.InsertPermission({
            'type': 'anyone',
            'value': 'reader',
            'role': 'reader'})

        print(f"Uploaded {file_name} to Google Drive. Link: {gfile['alternateLink']}")
        return gfile['alternateLink']
    except Exception as e:
        print(f"Error uploading {file_name} to Google Drive: {e}")
        return None

def upload_image_to_drive(file_path, file_name, target_folder_id=None):
    """
    Uploads an image file to Google Drive.
    If target_folder_id is provided, uploads to that folder.
    """
    global drive
    if drive is None:
        print("Google Drive not initialized. Cannot upload image.")
        return None

    try:
        file_metadata = {
            'title': file_name,
            'mimeType': 'image/jpeg'
        }
        if target_folder_id:
            file_metadata['parents'] = [{'id': target_folder_id}]

        gfile = drive.CreateFile(file_metadata)
        gfile.SetContentFile(file_path) # Corrected back to SetContentFile
        gfile.Upload()

        permission = gfile.InsertPermission({
            'type': 'anyone',
            'value': 'reader',
            'role': 'reader'})

        print(f"Uploaded {file_name} to Google Drive. Link: {gfile['alternateLink']}")
        return gfile['alternateLink']
    except Exception as e:
        print(f"Error uploading {file_name} to Google Drive: {e}")
        return None

# --- Helper Functions ---
def get_y_on_line(line_start, line_end, x_coord):
    x1, y1 = line_start
    x2, y2 = line_end
    if abs(x2 - x1) < 1:
        return y1
    slope = (y2 - y1) / (x2 - x1)
    y_intercept = y1 - slope * x1
    return slope * x_coord + y_intercept

def has_crossed_line(obj_id, line_name, point_prev, point_curr, line_start, line_end):
    line_y_at_prev_x = get_y_on_line(line_start, line_end, point_prev[0])
    line_y_at_curr_x = get_y_on_line(line_start, line_end, point_curr[0])
    crossed_from_above = (point_prev[1] < line_y_at_prev_x) and (point_curr[1] >= line_y_at_curr_x)
    return crossed_from_above

def get_box_points(bbox):
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    bottom_center_x = int((x1 + x2) / 2)
    bottom_center_y = y2
    return (center_x, center_y), (bottom_center_x, bottom_center_y)

# --- Global Tracking Variables ---
next_object_id = 0
tracked_objects = {}
frame_buffer = collections.deque() # Global frame buffer

# --- Tracking Logic (Modified to use global tracking variables) ---
def update_tracks(detections, current_frame_num, current_time, config_params, parking_zone_polygon_points):
    global next_object_id, tracked_objects
    current_tracks = list(tracked_objects.keys())
    unmatched_detections = []

    # Filter detections to exclude those within the parking zone (if defined)
    filtered_detections = []
    for det_info in detections:
        det_bottom_center = det_info['bottom_center']
        is_in_parking_zone = False
        if parking_zone_polygon_points is not None:
             is_in_parking_zone = cv2.pointPolygonTest(parking_zone_polygon_points, (int(det_bottom_center[0]), int(det_bottom_center[1])), False) >= 0
        
        if not is_in_parking_zone:
            filtered_detections.append(det_info)

    for det_idx, det_info in enumerate(filtered_detections):
        det_bbox = det_info['bbox']
        det_bottom_center = det_info['bottom_center']
        
        min_dist = float('inf')
        best_match_id = -1

        for track_id in current_tracks:
            track = tracked_objects[track_id]
            track_bottom_center = track['bottom_center']
            
            dist = distance.euclidean(det_bottom_center, track_bottom_center)

            if dist < min_dist and dist < config_params['max_distance_for_track']:
                min_dist = dist
                best_match_id = track_id
        
        if best_match_id != -1:
            tracked_objects[best_match_id]['last_pos_for_check'] = tracked_objects[best_match_id]['bottom_center']
            tracked_objects[best_match_id].update({
                'bbox': det_bbox,
                'bottom_center': det_bottom_center,
                'last_seen_frame': current_frame_num,
                'label': det_info['label'],           
                'confidence': det_info['confidence']    
            })
            current_tracks.remove(best_match_id)
        else:
            tracked_objects[next_object_id] = {
                'bbox': det_info['bbox'],
                'bottom_center': det_info['bottom_center'],
                'last_seen_frame': current_frame_num,
                'label': det_info['label'],       
                'confidence': det_info['confidence'], 
                'status': 'approaching_blue',
                'stopping_motion_start_time': None, # New: To track when minimal movement starts
                'tolerated_movement_frames_count': 0, # New: Counter for frames with tolerated movement
                'blue_crossed_timestamp': None,     # Remains for actual blue line crossing time (for violation)
                'violation_display_until_time': None,
                'last_pos_for_check': det_info['bottom_center'],
                'has_waited_in_green_zone': False, # Still a flag to mark a successful wait once
                'is_opposite_direction': False,
                'logged': False,
                'image_logged': False, # Flag for image logging
                'first_appearance_time': current_time, 
                'has_been_cleared_to_cross': False 
            }
            next_object_id += 1
    
    ids_to_remove = []
    for track_id in list(tracked_objects.keys()):
        # Remove tracks that haven't been seen for too long AND are not currently in violation display state
        if (tracked_objects[track_id]['violation_display_until_time'] is None or \
            current_time >= tracked_objects[track_id]['violation_display_until_time']) and \
            (current_frame_num - tracked_objects[track_id]['last_seen_frame'] > config_params['max_frames_to_skip']):
            ids_to_remove.append(track_id) # Collect IDs to remove
            
    for track_id in ids_to_remove: # Remove after iteration
        del tracked_objects[track_id]

    return tracked_objects


# --- Main processing function for a single frame ---
def process_frame(video_source_obj, is_vlc_source, frame_count, current_time,
                  blue_line_start, blue_line_end,
                  green_line1_start, green_line2_start,
                  green_line1_end, green_line2_end,
                  green_zone_polygon_points, parking_zone_polygon_points,
                  config_params, output_video_writer, run_yolo_inference, last_known_detections):
    global pixels, vlc_video_dims, frame_buffer # Ensure global declarations for modified globals

    frame = None
    if is_vlc_source:
        with pixels_lock:
            if pixels is not None and vlc_video_dims['width'] > 0 and vlc_video_dims['height'] > 0 and vlc_video_dims['pitch'] > 0:
                try:
                    # Attempt to get the latest frame from VLC's memory buffer
                    frame_raw_2d = pixels.reshape((vlc_video_dims['height'], vlc_video_dims['pitch']))
                    frame_sliced = frame_raw_2d[:, :vlc_video_dims['width'] * 4]
                    frame_bgrx = cv2.resize(frame_sliced.reshape((vlc_video_dims['height'], vlc_video_dims['width'], 4)), 
                                            (vlc_video_dims['width'], vlc_video_dims['height']))
                    frame = cv2.cvtColor(frame_bgrx, cv2.COLOR_BGRA2BGR) 
                except ValueError as e:
                    print(f"Error reshaping/converting VLC frame buffer: {e}. Dims: W:{vlc_video_dims['width']}, H:{vlc_video_dims['height']}, P:{vlc_video_dims['pitch']}")
                    return None # Return None on error
            else:
                return None # No frame available from VLC buffer either
    else: # Use OpenCV VideoCapture
        ret, frame = video_source_obj.read()
        if not ret:
            # print("Warning: End of stream or error reading frame from OpenCV. Skipping.") # Suppress for continuous streams
            return None # No frame available from OpenCV source
    
    if frame is None:
        return None
    
    # --- Defensive check for frame validity before copying ---
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        print("Warning: Received invalid or empty frame for processing. Skipping.")
        return None

    # --- Frame Buffering (for clip saving, even if not uploaded, useful for debug) ---
    frame_buffer.append((current_time, frame.copy())) # Keep frame copy for buffering
    while frame_buffer and (current_time - frame_buffer[0][0] > config_params['buffer_duration_seconds']):
        frame_buffer.popleft()

    current_frame_detections = []
    if run_yolo_inference:
        # Perform YOLO inference only if explicitly told to for this frame
        results = model(frame, verbose=False, conf=config_params['conf_threshold'])
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                label = model.names[cls_id]

                if label in vehicle_class_names:
                    _, bottom_center = get_box_points((x1, y1, x2, y2))
                    current_frame_detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'label': label,
                        'confidence': conf,
                        'bottom_center': bottom_center
                    })
    else:
        # If not running YOLO, reuse the last known detections for tracking
        current_frame_detections = last_known_detections

    active_tracks = update_tracks(current_frame_detections, frame_count, current_time, config_params, parking_zone_polygon_points)

    # Draw parking zone polygon (BEFORE other drawings so it's a background element)
    if parking_zone_polygon_points is not None:
        cv2.polylines(frame, [parking_zone_polygon_points], True, (255, 100, 0), 2)
        min_x = min(p[0] for p in parking_zone_polygon_points)
        min_y = min(p[1] for p in parking_zone_polygon_points)
        cv2.putText(frame, "Parking Zone", (min_x, min_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

    image_link_for_current_frame = ""
    # --- Draw annotations for all tracked objects and apply logic ---
    for obj_id, track in active_tracks.items():
        bbox = track['bbox']
        bottom_center = track['bottom_center']
        status = track['status']

        x1, y1, x2, y2 = bbox
        label = track['label']
        prev_bottom_center = track['last_pos_for_check']

        # --- Direction Filtering Logic ---
        if not track['is_opposite_direction']:
            y_movement = bottom_center[1] - prev_bottom_center[1]
            if abs(y_movement) > config_params['direction_movement_threshold_pixels']:
                if y_movement < 0: # Moving upwards means opposite direction across horizontal lines
                    track['is_opposite_direction'] = True
        
        # --- Visuals for Opposite Direction Vehicles ---
        if track['is_opposite_direction']:
            box_color = (180, 180, 180) # Grey for opposite direction
            status_text = f"{label} (Opposite Dir)"
            if x2 > x1 and y2 > y1: # Ensure bbox is valid before drawing
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame, status_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                cv2.circle(frame, bottom_center, 6, (100, 100, 100), -1)
                cv2.circle(frame, prev_bottom_center, 3, (70, 70, 70), -1)
            continue # Skip rest of rule logic for opposite direction vehicles

        # --- Rule Logic for Normal Direction Vehicles ---
        is_in_green_zone = cv2.pointPolygonTest(green_zone_polygon_points, (int(bottom_center[0]), int(bottom_center[1])), False) >= 0
        movement_this_frame = distance.euclidean(bottom_center, prev_bottom_center)
        is_physically_across_blue_now = bottom_center[1] >= get_y_on_line(blue_line_start, blue_line_end, bottom_center[0])
        
        # Conditions for "stopping motion" (minimal movement)
        is_currently_minimal_movement = movement_this_frame < config_params['static_movement_threshold_pixels']

        if status == 'approaching_blue':
            # Check for immediate violation if it crosses the blue line prematurely
            if has_crossed_line(obj_id, "Blue", prev_bottom_center, bottom_center, blue_line_start, blue_line_end):
                track['status'] = 'violated_blue_early'
                track['violation_display_until_time'] = current_time + config_params['violation_display_duration']
                track['blue_crossed_timestamp'] = current_time # This marks the time of violation
                print(f"ID {obj_id}: DEBUG: VIOLATION - Directly crossed blue line from 'approaching_blue'. STATUS SET TO: {track['status']}")
            
            # If not a violation yet, check if it starts "stopping motion" in green zone
            elif is_in_green_zone and is_currently_minimal_movement:
                track['stopping_motion_start_time'] = current_time # Start the stopping timer
                track['tolerated_movement_frames_count'] = 0 # Reset tolerance counter
                track['status'] = 'waiting_for_blue' # Transition to waiting state
                print(f"ID {obj_id}: DEBUG: Entered 'stopping motion' in green zone at {current_time:.2f}. Status: {track['status']}")

        elif status == 'waiting_for_blue':
            # Conditions to *maintain* 'waiting_for_blue' status:
            if is_in_green_zone:
                if is_currently_minimal_movement:
                    track['tolerated_movement_frames_count'] = 0 # Reset tolerance if truly minimal
                else: # Movement is above static_movement_threshold_pixels, but still in green zone
                    track['tolerated_movement_frames_count'] += 1
                
                # Check if tolerance limit is exceeded
                if track['tolerated_movement_frames_count'] > config_params['max_tolerated_frames_for_wait']:
                    track['stopping_motion_start_time'] = None # Reset timer
                    track['tolerated_movement_frames_count'] = 0 # Reset counter
                    track['status'] = 'approaching_blue' # Revert to approaching due to sustained movement
                    print(f"ID {obj_id}: DEBUG: Left 'waiting_for_blue' state (moved too much, tolerance exceeded). Reverted to 'approaching_blue'.")
                else: # Still within tolerance or truly minimal movement in green zone
                    time_in_stopping_motion = current_time - track['stopping_motion_start_time']
                    
                    # If waited long enough, clear to cross
                    if time_in_stopping_motion >= config_params['stop_duration_seconds']:
                        track['status'] = 'cleared_to_cross_blue'
                        track['has_been_cleared_to_cross'] = True
                        track['violation_display_until_time'] = None # Clear any past violation display if applicable
                        print(f"ID {obj_id}: DEBUG: Cleared to cross blue line (stopped/minimal movement for {config_params['stop_duration_seconds']}s). STATUS SET TO: {track['status']}")
                    
                    # Check for violation if it crosses blue line while still in 'waiting' state and not cleared
                    # This must come after checking for clearance, to ensure clearance takes precedence
                    if is_physically_across_blue_now and not track['has_been_cleared_to_cross']:
                        # This implies it moved across before the timer elapsed
                        track['status'] = 'violated_blue_early'
                        track['violation_display_until_time'] = current_time + config_params['violation_display_duration']
                        track['blue_crossed_timestamp'] = current_time # Mark the time of violation
                        print(f"ID {obj_id}: DEBUG: VIOLATION - Crossed blue line from 'waiting_for_blue' before cleared. STATUS SET TO: {track['status']}")
            else: # Vehicle left green zone while waiting
                track['stopping_motion_start_time'] = None # Reset timer
                track['tolerated_movement_frames_count'] = 0 # Reset counter
                track['status'] = 'approaching_blue' # Revert to approaching
                print(f"ID {obj_id}: DEBUG: Left 'waiting_for_blue' state (left green zone). Reverted to 'approaching_blue'.")


        elif status == 'cleared_to_cross_blue':
            # Vehicle has fulfilled the wait condition. Now, detect if it crosses the blue line.
            if has_crossed_line(obj_id, "Blue", prev_bottom_center, bottom_center, blue_line_start, blue_line_end):
                track['status'] = 'crossed_blue_legally'
                track['violation_display_until_time'] = None # Clear any lingering red box
                print(f"ID {obj_id}: DEBUG: Crossed blue line legally. STATUS SET TO: {track['status']}")
            
        elif status == 'violated_blue_early':
            # This state persists for violation_display_duration. No new rule checks needed here unless for re-violation
            pass

        # --- Determine Bounding Box Color for Current Frame ---
        box_color = (240, 240, 240) # Default light gray

        if track['violation_display_until_time'] is not None and current_time < track['violation_display_until_time']:
            box_color = (0, 0, 255) # Red for violation
        elif status == 'approaching_blue':
            box_color = (240, 240, 240) # Light gray
        elif status == 'waiting_for_blue':
            box_color = (0, 165, 255) # Orange for waiting
        elif status == 'cleared_to_cross_blue':
            box_color = (0, 255, 0) # Green for cleared
        elif status == 'crossed_blue_legally':
            box_color = (240, 240, 240) # Back to light gray after legal cross
        # No else needed, box_color is already default for unhandled states

        # --- Draw bounding box and status text on the current frame ---
        if x2 > x1 and y2 > y1: # Ensure bbox is valid before drawing
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
            status_text = f"{label} ({status.replace('_',' ').title()})"
            
            # Display time remaining for waiting vehicles
            if status == 'waiting_for_blue' and track['stopping_motion_start_time'] is not None:
                time_elapsed = current_time - track['stopping_motion_start_time']
                time_remaining = max(0, config_params['stop_duration_seconds'] - time_elapsed)
                status_text += f" (Wait: {time_remaining:.1f}s)"

            cv2.putText(frame, status_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            cv2.circle(frame, bottom_center, 6, (0, 255, 255), -1) # Yellow circle at bottom center
            cv2.circle(frame, prev_bottom_center, 3, (128, 0, 128), -1) # Purple circle for previous position
        else:
            pass # Invalid bbox, skip drawing

        # --- Capture and Upload Violation Image (as soon as the box turns red, only once per violation) ---
        if track['status'] == 'violated_blue_early' and not track['image_logged']:
            print(f"ID {obj_id}: VIOLATION detected. Attempting to capture image.")
            image_logged_success = False
            
            violation_images_folder_id = None
            if drive:
                violation_images_folder_id = get_or_create_gdrive_folder("Violation_Images", parent_folder_id=config_params['gdrive_root_folder_id'])
            else:
                print("Google Drive not initialized, skipping image upload.")

            if violation_images_folder_id:
                temp_image_filename = f"VIOLATION_{label}_ID{obj_id}_{datetime.datetime.fromtimestamp(current_time).strftime('%Y%m%d_%H%M%S')}.jpg"
                temp_image_path = os.path.join(config_params['temp_violation_images_dir'], temp_image_filename)
                
                # IMPORTANT: Create a copy of the *current annotated frame* for the image capture
                # This ensures the saved image includes the red box and "Violation" text.
                frame_for_image_capture = frame.copy() 
                
                try:
                    cv2.imwrite(temp_image_path, frame_for_image_capture)
                    print(f"Temporary local image saved: {temp_image_path}")
                    image_link_for_current_frame = upload_image_to_drive(temp_image_path, temp_image_filename, violation_images_folder_id)
                    if image_link_for_current_frame:
                        image_logged_success = True
                        print(f"Image uploaded successfully: {image_link_for_current_frame}")
                    else:
                        image_link_for_current_frame = "Error Uploading Image"
                        print("Image upload failed.")
                except Exception as e:
                    print(f"Error saving or uploading image {temp_image_path}: {e}")
                    image_link_for_current_frame = "Error: Image save/upload failed"
                finally:
                    # Clean up temporary image file regardless of upload success
                    if os.path.exists(temp_image_path):
                        try:
                            os.remove(temp_image_path)
                            print(f"Temporary local image deleted: {temp_image_path}")
                        except OSError as e:
                            print(f"Error deleting temporary image {temp_image_path}: {e}")

            if image_logged_success:
                track['image_logged'] = True


        # --- Handle Logging (Video uploads commented out) ---
        # Log for vehicles entering the main area (not opposite direction)
        if not track['logged'] and not track['is_opposite_direction'] and \
           (track['status'] == 'crossed_blue_legally' or \
            track['status'] == 'violated_blue_early'):
            
            track['logged'] = True
            event_timestamp = current_time
            
            target_drive_folder_id = None
            if drive:
                current_violation_status_for_logic = "Yes" if track['status'] == 'violated_blue_early' else "No"
                
                if current_violation_status_for_logic == "Yes":
                    violations_folder_id = get_or_create_gdrive_folder("Violations", parent_folder_id=config_params['gdrive_root_folder_id'])
                    target_drive_folder_id = violations_folder_id
                else:
                    non_violations_root_folder_id = get_or_create_gdrive_folder("Non_Violations_By_Type", parent_folder_id=config_params['gdrive_root_folder_id'])
                    vehicle_type_folder_id = get_or_create_gdrive_folder(label.capitalize(), parent_folder_id=non_violations_root_folder_id)
                    target_drive_folder_id = vehicle_type_folder_id
            else:
                print("Google Drive not initialized, cannot determine target folder for upload.")

            temp_clip_filename = f"{label}_ID{obj_id}_{datetime.datetime.fromtimestamp(event_timestamp).strftime('%Y%m%d_%H%M%S')}.mp4"
            temp_clip_path = os.path.join(config_params['temp_output_clips_dir'], temp_clip_filename)
            
            video_link = "" # Video upload commented out for efficiency. Can re-enable if needed.

            timestamp_log = datetime.datetime.fromtimestamp(event_timestamp).strftime("%Y-%m-%d %H:%M:%S")
            stop_status_log = "Stopped" if track['status'] == 'crossed_blue_legally' else "Doesn't stop"
            wait_status_log = "Yes" if track['status'] == 'crossed_blue_legally' else "No"
            violation_log = "No" if track['status'] == 'crossed_blue_legally' else "Yes"
            
            print(f"DEBUG: Attempting to log compliant vehicle (ID: {obj_id}, Status: {track['status']}) to sheet.")
            log_vehicle_data(timestamp_log, config_params['location'], label, "In", stop_status_log, wait_status_log, violation_log, 
                             video_link, 
                             image_link_for_current_frame if violation_log == "Yes" else "")


        # Log for vehicles moving in the opposite direction
        elif not track['logged'] and track['is_opposite_direction']:
            
            track['logged'] = True
            event_timestamp = current_time

            target_drive_folder_id = None
            if drive:
                non_violations_root_folder_id = get_or_create_gdrive_folder("Non_Violations_By_Type", parent_folder_id=config_params['gdrive_root_folder_id'])
                vehicle_type_folder_id = get_or_create_gdrive_folder(label.capitalize(), parent_folder_id=non_violations_root_folder_id)
                target_drive_folder_id = vehicle_type_folder_id
            else:
                print("Google Drive not initialized, cannot determine target folder for upload.")

            temp_clip_filename = f"{label}_ID{obj_id}_OUT_{datetime.datetime.fromtimestamp(event_timestamp).strftime('%Y%m%d_%H%M%S')}.mp4"
            temp_clip_path = os.path.join(config_params['temp_output_clips_dir'], temp_clip_filename)
            
            video_link = "" # Video upload commented out for efficiency.

            timestamp_log = datetime.datetime.fromtimestamp(event_timestamp).strftime("%Y-%m-%d %H:%M:%S")
            
            print(f"DEBUG: Attempting to log opposite direction vehicle (ID: {obj_id}, Status: {track['status']}) to sheet.")
            log_vehicle_data(timestamp_log, config_params['location'], label, "Out", "N/A", "N/A", "No", 
                             video_link, 
                             "")
    
    # --- Draw the custom detection lines on the current frame ---
    cv2.line(frame, blue_line_start, blue_line_end, config_params['blue_color'], config_params['line_thickness'])
    cv2.line(frame, green_line1_start, green_line1_end, config_params['green_color'], config_params['green_line_thickness'])
    cv2.line(frame, green_line2_start, green_line2_end, config_params['green_color'], config_params['green_line_thickness'])

    # Write the processed frame to the output video file (if enabled in config and writer is initialized)
    if output_video_writer is not None:
        output_video_writer.write(frame)

    return frame # Return the processed frame for GUI display

# --- Main execution block for headless operation ---
def run_headless_processor(config_params):
    # Initialize VLC components
    vlc_instance = None
    vlc_player = None
    cap = None # This will be the OpenCV VideoCapture object if not using VLC
    frame_width = 0
    frame_height = 0
    stream_fps = 0 

    # Determine if we are using VLC (RTSP) or OpenCV (local file/simple RTSP)
    use_vlc = config_params['video_source'].startswith("rtsp://")

    if use_vlc:
        # Set VLC plugin path for proper functionality on Windows
        # Adjust this path if your VLC installation is in a different directory!
        os.environ['VLC_PLUGIN_PATH'] = 'C:\\Program Files\\VideoLAN\\VLC\\plugins'
        print(f"Set VLC_PLUGIN_PATH to: {os.environ['VLC_PLUGIN_PATH']}")

        # VLC Instance with improved caching and live stream options
        vlc_instance = vlc.Instance([
            '--rtsp-tcp',               # Force TCP for RTSP (more reliable than UDP over lossy networks)
            '--network-caching=3000',   # Increased network caching to 3000ms (3 seconds)
            '--live-caching=3000',      # Specific caching for live streams
            '--sout-mux-caching=1000',  # Increased caching for stream output
            '--clock-synchro=0',        # Disable clock synchronization to reduce frame drops due to timing issues
            '--file-caching=3000',      # Increased general file caching
            '--drop-late-frames',       # Explicitly drop late frames instead of showing them
            '--skip-frames',            # Skip frames to catch up if lagging
            '--ignore-config',          # Ignore user config, rely on command line args
            '--no-video-title-show',    # Do not display title in video
            '--verbose=0',              # Set verbose to 0 (silent) to suppress H.264 errors unless critical
            '--no-stats',               # Disable statistics output
            '--no-auto-preparse',       # Don't auto-preparse media
        ])
        
        if not vlc_instance:
            print("Error: Could not create VLC instance. Please check VLC installation and plugin path.")
            return # Exit if VLC instance fails

        vlc_player = vlc_instance.media_player_new()
        
        if not vlc_player:
            print("Error: Could not create VLC media player.")
            return # Exit if VLC media player fails

        # Configure VLC to render frames to memory
        vlc_player.video_set_format('RV32', vlc_video_dims['width'], vlc_video_dims['height'], vlc_video_dims['pitch'])
        vlc_player.video_set_callbacks(video_lock_cb, video_unlock_cb, video_display_cb, ctypes.c_void_p(0))

        media = vlc_instance.media_new(config_params['video_source'])
        vlc_player.set_media(media)
        vlc_player.audio_set_mute(True) # Mute audio for video processing
        
        # Start playing with a longer warm-up to fill buffers
        print(f"VLC Player attempting to play: {config_params['video_source']}")
        vlc_player.play()
        # Initial longer sleep to allow VLC to buffer sufficiently
        time.sleep(7) # Increased initial buffer time to 7 seconds

        # Attempt to get actual stream dimensions/FPS after a sufficient delay
        try:
            width = vlc_player.video_get_width()
            height = vlc_player.video_get_height()
            if width > 0 and height > 0:
                frame_width = width
                frame_height = height
                stream_fps = config_params['yolo_inference_fps'] # Still use configured FPS as target
                print(f"RTSP stream detected resolution: {frame_width}x{frame_height}. Processing at configured YOLO FPS: {config_params['yolo_inference_fps']}")
            else:
                print("Could not get RTSP stream resolution from VLC. Using default 1280x720 and configured YOLO FPS.")
                frame_width = 1280
                frame_height = 720
                stream_fps = config_params['yolo_inference_fps']
        except Exception as e:
            print(f"Error getting VLC stream info: {e}. Using default 1280x720 and configured YOLO FPS.")
            frame_width = 1280
            frame_height = 720
            stream_fps = config_params['yolo_inference_fps']

    else: # For local video files (or if VLC is not preferred for RTSP)
        cap = cv2.VideoCapture(config_params['video_source'])
        if not cap.isOpened():
            print(f"Error: Could not open video stream from {config_params['video_source']}. Please check the URL/path and stream availability.")
            return # Exit if video stream cannot be opened
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        stream_fps = cap.get(cv2.CAP_PROP_FPS)
        if stream_fps <= 0: 
            stream_fps = 30 # Default to 30 FPS if not detectable
        print(f"Status: OpenCV VideoCapture opened local video. Resolution: {frame_width}x{frame_height}, Actual FPS: {stream_fps:.2f}")

    # Calculate YOLO inference skip interval based on actual stream FPS and desired YOLO FPS
    if config_params['yolo_inference_fps'] > 0 and stream_fps > 0:
        yolo_frame_skip_interval = max(1, int(round(stream_fps / config_params['yolo_inference_fps'])))
    else:
        yolo_frame_skip_interval = 1 
    print(f"YOLO inference will run every {yolo_frame_skip_interval} frames (effective YOLO FPS: {stream_fps / yolo_frame_skip_interval:.2f}).")


    # Set up line and zone coordinates
    blue_line_start = config_params['blue_line_start']
    blue_line_end = config_params['blue_line_end']
    green_line1_start = config_params['green_line1_start']
    green_line2_start = config_params['green_line2_start']
    green_line1_end = blue_line_start # Derived
    green_line2_end = blue_line_end # Derived

    green_zone_polygon_points = np.array([
        green_line1_start,
        green_line2_start,
        blue_line_end,
        blue_line_start
    ], np.int32)

    parking_zone_polygon_points = None
    if config_params['parking_zone_point1'] is not None:
        parking_zone_polygon_points = np.array([
            config_params['parking_zone_point1'],
            config_params['parking_zone_point2'],
            config_params['parking_zone_point3'],
            config_params['parking_zone_point4']
        ], np.int32)
    
    # Initialize Google Sheet and Drive
    initialize_google_sheet_logger(config_params['google_sheet_id'])
    initialize_google_drive(config_params['gdrive_root_folder_id'])

    # Initialize output video writer
    output_video_writer = None
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video_writer = cv2.VideoWriter(config_params['output_processed_video_path'], fourcc, stream_fps, (frame_width, frame_height))
        if not output_video_writer.isOpened():
            print(f"Warning: Could not open video writer for output video at {config_params['output_processed_video_path']}. Processed video will not be saved.")
            output_video_writer = None
        else:
            print(f"Saving processed video to: {config_params['output_processed_video_path']}")
    except Exception as e:
        print(f"Error initializing output video writer: {e}. Processed video will not be saved.")
        output_video_writer = None

    frame_count = 0
    last_yolo_detections = [] # To store detections from the last YOLO run
    
    # Track the last time a VLC state warning was printed to avoid flooding the console
    last_vlc_warning_time = time.time()
    vlc_warning_cooldown = 10 # seconds

    try:
        while True:
            current_time = time.time()
            
            run_yolo_on_this_frame = (frame_count % yolo_frame_skip_interval == 0)

            if use_vlc:
                vlc_state = vlc_player.get_state()
                # Check for critical VLC states
                if vlc_state not in [vlc.State.Playing, vlc.State.Buffering]:
                    if time.time() - last_vlc_warning_time > vlc_warning_cooldown:
                        print(f"Warning: VLC player is not in playing or buffering state ({str(vlc_state).split('.')[-1]}). Stream might be interrupted or ended.")
                        last_vlc_warning_time = time.time()
                    
                    # If VLC is paused, ended, or stopped, try to restart it
                    if vlc_state in [vlc.State.Ended, vlc.State.Error, vlc.State.Stopped]:
                        print(f"VLC stream ended or encountered an error. State: {str(vlc_state).split('.')[-1]}. Attempting to re-open stream in 5 seconds...")
                        vlc_player.stop() # Ensure it's stopped
                        time.sleep(5) # Wait before retrying
                        media = vlc_instance.media_new(config_params['video_source'])
                        vlc_player.set_media(media)
                        vlc_player.play()
                        time.sleep(3) # Give time to re-buffer
                        continue # Skip to next loop iteration, attempt to get frame on next iteration
                
                # Dynamically update VLC video dims if they change (e.g. stream changes resolution)
                with pixels_lock:
                    width = vlc_player.video_get_width()
                    height = vlc_player.video_get_height()
                    if width > 0 and height > 0 and (vlc_video_dims['width'] != width or vlc_video_dims['height'] != height):
                        vlc_video_dims['width'] = width
                        vlc_video_dims['height'] = height
                        vlc_video_dims['pitch'] = width * 4 # Assume RV32
                        print(f"VLC stream resolution changed to: {width}x{height}")
                        # Reallocate pixels buffer if dimensions changed significantly
                        global pixels
                        pixels = np.zeros((vlc_video_dims['height'], vlc_video_dims['pitch']), dtype=np.uint8)

                # Process frame using VLC as the source
                processed_frame = process_frame(
                    vlc_player, True, frame_count, current_time,
                    blue_line_start, blue_line_end,
                    green_line1_start, green_line2_start,
                    green_line1_end, green_line2_end,
                    green_zone_polygon_points, parking_zone_polygon_points,
                    config_params, output_video_writer,
                    run_yolo_on_this_frame, last_yolo_detections
                )

            else: # Use OpenCV VideoCapture for local files or direct RTSP (if configured this way)
                # Process frame using OpenCV cap as the source
                processed_frame = process_frame(
                    cap, False, frame_count, current_time,
                    blue_line_start, blue_line_end,
                    green_line1_start, green_line2_start,
                    green_line1_end, green_line2_end,
                    green_zone_polygon_points, parking_zone_polygon_points,
                    config_params, output_video_writer,
                    run_yolo_on_this_frame, last_yolo_detections
                )
            
            if processed_frame is None:
                # If stream ends or has a critical error, and we couldn't recover
                if use_vlc and vlc_player.get_state() not in [vlc.State.Playing, vlc.State.Buffering]:
                    print("VLC stream seems to be permanently interrupted. Exiting.")
                    break
                elif not use_vlc and not cap.isOpened(): # For local file, if cap is closed, it's end of file
                    print("Local video stream ended. Exiting.")
                    break
                # If it's a transient issue (e.g., VLC still trying to buffer, or OpenCV temporarily drops a frame), wait a bit
                time.sleep(0.05) # Small delay before trying to get another frame
                continue # Try again
            
            # If YOLO ran on this frame, update last_yolo_detections
            if run_yolo_on_this_frame: # Use the flag directly
                # Extract detections from active_tracks that were *just* updated by YOLO
                current_yolo_dets = []
                for obj_id, track in tracked_objects.items():
                    # Only consider objects that were seen in the current frame and are not opposite direction
                    if track['last_seen_frame'] == frame_count and not track['is_opposite_direction']:
                        current_yolo_dets.append({
                            'bbox': track['bbox'],
                            'label': track['label'],
                            'confidence': track['confidence'],
                            'bottom_center': track['bottom_center']
                        })
                last_yolo_detections = current_yolo_dets
            
            frame_count += 1
            # Adjust sleep time to control the overall processing loop FPS to match stream FPS
            time_spent = time.time() - current_time
            sleep_time = (1 / stream_fps) - time_spent
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("Processing interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")
    finally:
        print("Application stopping. Releasing resources...")
        if use_vlc and vlc_player:
            vlc_player.stop()
            print("VLC player stopped.")
        elif not use_vlc and cap and cap.isOpened():
            cap.release()
            print("OpenCV VideoCapture released.")
        if output_video_writer is not None:
            output_video_writer.release()
            print(f"Processed video saved to: {config_params['output_processed_video_path']}")
        cv2.destroyAllWindows()
        print("Resources released. Application exited.")


if __name__ == "__main__":
    run_headless_processor(config_params)
