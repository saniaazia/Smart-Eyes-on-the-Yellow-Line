Step1: Download mediamtx, extract it.

Step2: Under mediamtx.yml inset your RTSP link

Step3: Start mediamtx.exe by running startup.bat

Step 4: Run the main.py, make necessary changes in the code as directed by comments.

Step4.1: Install the necessary libraries:
python -m pip install opencv-python numpy scipy ultralytics PyQt5 python-vlc gspread oauth2client PyDrive

Step4.2: Create a google sheet and follow the steps below:
        1. Go to Google Cloud Console.

        2. Create a new project (or select an existing one).

        3. Enable the Google Sheets API and Google Drive API.

        4. Go to APIs & Services > Credentials.

        5. Click “Create Credentials” → Service Account.

        6. After creation, click the service account and go to the “Keys” tab.

        7. Add a new key → Choose JSON → Download the file (this is your key).

        8. Rename the file to something like creds.json and place it in your project directory.

        9. Open your Google Sheet. Click Share. Share with the service account email, something like: my-service-account@my-project.iam.gserviceaccount.com

        10. Give it Editor access.

    Step4.3: Make sure VLC is installed and both VSC and VLC are of the latest version.

Step5: Run the main file. Hope it works for you!



