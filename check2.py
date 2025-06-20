import cv2
import numpy as np

def stream_with_lines(rtsp_url):
    """
    Streams an RTSP video feed and overlays predefined blue and green lines.
    Press 'ESC' to exit the stream.
    """
    # Define line coordinates
    # Blue line (main reference line)
    # blue_line_start = (80, 500)
    # blue_line_end = (1265, 646)

    # # Green lines (defining a zone relative to the blue line)
    # # Based on your previous code structure, these might represent a zone entry.
    # # green_line1_end is derived from blue_line_start
    # # green_line2_end is derived from blue_line_end
    # green_line1_start = (512, 225)
    # green_line2_start = (1269, 302)
    blue_line_start = (273, 680)
    blue_line_end = (1171, 660)
# Define the start points for your green lines.
# Note: green_line1_end and green_line2_end automatically use blue_line_start/end in the code.
    green_line1_start = (82, 327)
    green_line2_start = (725,299)
    green_line1_end = blue_line_start # Assuming it connects to blue_line_start
    green_line2_end = blue_line_end   # Assuming it connects to blue_line_end

    # Define colors and thickness for the lines (BGR format)
    blue_color = (255, 0, 0)   # Blue line
    green_color = (0, 255, 0)  # Green lines
    line_thickness = 3         # Thickness for all lines

    # Initialize video capture from the RTSP stream
    cap = cv2.VideoCapture(rtsp_url)

    # Check if the stream opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video stream from {rtsp_url}.")
        print("Please ensure the URL is correct, the camera is accessible, and OpenCV's FFmpeg backend supports the stream.")
        return

    print(f"Streaming from: {rtsp_url}")
    print("Press 'ESC' key to exit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Stream ended or error reading frame. Exiting...")
            break

        # Draw the blue line
        cv2.line(frame, blue_line_start, blue_line_end, blue_color, line_thickness)

        # Draw the two green lines
        cv2.line(frame, green_line1_start, green_line1_end, green_color, line_thickness)
        cv2.line(frame, green_line2_start, green_line2_end, green_color, line_thickness)

        # Display the frame
        cv2.imshow("RTSP Stream with Lines", frame)

        # Exit if 'ESC' key is pressed
        if cv2.waitKey(1) & 0xFF == 27: # 27 is the ASCII for ESC key
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace with your actual RTSP URL (e.g., from your camera or mediamtx)
    # Example using localhost mediamtx stream: "rtsp://localhost:8554/cam1"
    # Example using your direct camera stream (if it works with OpenCV): "rtsp://admin:Secure17@172.17.210.184:554/Streaming/Channels/101"
    
    # Using your original camera URL from the config.ini, assuming it will work here for simplicity
    # If this still gives an error, consider using mediamtx as a proxy as discussed previously,
    # and update the rtsp_url to point to mediamtx's output (e.g., "rtsp://localhost:8554/cam1").
    
    # NOTE: If the original camera RTSP URL still causes issues with cv2.VideoCapture,
    # it is highly recommended to use `mediamtx` as a proxy as discussed in previous turns.
    # In that case, update the rtsp_url below to your mediamtx output (e.g., "rtsp://localhost:8554/cam1").
    
    rtsp_stream_url = "rtsp://localhost:8554/cam1"
    
    stream_with_lines(rtsp_stream_url)
