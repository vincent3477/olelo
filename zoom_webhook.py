"""
References:
https://chatgpt.com/share/69a37f14-bd48-8001-944f-434d019dc6d5 -- Validating a URL for the first time. 
https://chatgpt.com/share/69a4d446-bf28-8001-a546-9303bc2ca6e4 -- Learning how to validate and download zoom recordings using webhooks.
"""

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
import requests
import hashlib
import uvicorn
import base64
import os
import hmac
import time


app = FastAPI(title = "Olelo Notetaker")

ZOOM_WEBHOOK_SECRET = os.getenv("ZOOM_WEBHOOK_SECRET") # set this in your env
CLIENT_ID = os.getenv("ZOOM_CLIENT_ID")
ACCOUNT_ID = os.getenv("ZOOM_ACCOUNT_ID")
CLIENT_SECRET = os.getenv("ZOOM_CLIENT_SECRET")

RETRYABLE_ERROR_CODES = { 408, 429, 500, 502, 503, 504}
REDIRECT_CODES = {301, 302, 303, 307, 308}

from fastapi.responses import JSONResponse

app = FastAPI()


def get_access_token():
    ## GET AN ACCESS TOKEN
    credentials = f"{CLIENT_ID}:{CLIENT_SECRET}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()

    headers = {"Authorization": f"Basic {encoded_credentials}"}

    url = f"https://zoom.us/oauth/token?grant_type=account_credentials&account_id={ACCOUNT_ID}"

    response = requests.post(url, headers=headers)
    response.raise_for_status()

    return response.json()["access_token"]


def get_audio_file(body):
    download_token = body["download_token"]
    headers = {"Authorization": f"Bearer {download_token}"}
    recording_files = body["payload"]["object"]['recording_files']
    
    print(body)

    # Download each recording file
    for f in recording_files:
        record_name = f["id"]
        extension = f["file_extension"].lower()
        location = f["download_url"]
        if extension == "m4a" or extension == "mp3":
            for attempt in range(5):
                try:  
                    download_req_response = requests.get(location, headers=headers, stream=True, allow_redirects=False)
                    
                    if download_req_response in RETRYABLE_ERROR_CODES and download_req_response not in REDIRECT_CODES:
                        raise RuntimeError(f"Error {download_req_response.status_code} while trying to request download")


                    if download_req_response.status_code in REDIRECT_CODES:
                        location = download_req_response.headers.get("Location")
                        if not location:
                            raise RuntimeError(f"No location header: {download_req_response.status_code}")
                        continue


                    download_req_response.raise_for_status()

                    record_filename = f"{record_name}.{extension}"

                    with open(record_filename, "wb") as file_out:
                        for chunk in download_req_response.iter_content(chunk_size=8192):
                            if chunk:
                                file_out.write(chunk)
                    
                    break # Finish execution when everything went successfully.
                
                except RuntimeError as e:
                    print(e)
                    time.sleep(5)
                    if attempt == 4:
                        print("This is the last retry. Quitting.")
                        break

                except Exception as e:
                    print(e)
                    return 1


    return 0




@app.post("/webhook")
async def zoom_webhook(request: Request, background_tasks: BackgroundTasks):

    body = await request.json()
    if body.get("event") == "endpoint.url_validation": ## WEBHOOK VALIDATION

        key = ZOOM_WEBHOOK_SECRET.encode()

        plain_token = body["payload"]["plainToken"]
        message = plain_token.encode()

        encrypted_token = hmac.new(key, message, hashlib.sha256).hexdigest()
        # hash digest is the fixed-size binary output from a crypto hash function (but this time we are doing a hex string format)

        return {"plainToken":plain_token, "encryptedToken":encrypted_token}
    
    if body.get("event") == "recording.completed": ## MEETING RECORDING RECEIVED
      
        background_tasks.add_task(get_audio_file, body)

        return {"ok": True}
            
        







    

    


   



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port = 8080, log_level = "info")