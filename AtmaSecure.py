import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace
from twilio.rest import Client
import pyttsx3
import time
import os
import pandas as pd
from streamlit_geolocation import streamlit_geolocation

# --- CONFIG ---
LOG_FILE = "event_log.csv"
PHOTO_DIR = "alert_photos"
if not os.path.exists(PHOTO_DIR):
    os.makedirs(PHOTO_DIR)
contacts = [
    "whatsapp:+918958034551",
    "whatsapp:+917351271327"
    # Add your WhatsApp contacts
    # Add more WhatsApp contacts as needed
]
TWILIO_WHATSAPP_FROM = "whatsapp:+14155238886"
TWILIO_SID = "AC080cd6c224d08105623d421fcdb396c9"
TWILIO_TOKEN = "3033b9ff25b70e3e9fd1e2d47984f9d4"
WOMEN_HELPLINE_NUMBER = "1091"  # Indian Women Helpline Example

# --- Hand Detection ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
def detect_open_palm(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    return bool(results.multi_hand_landmarks)

# --- Emotion Detection ---
def detect_fear(frame):
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)
        if isinstance(result, list):
            detected_emotion = result[0]['dominant_emotion']
        else:
            detected_emotion = result['dominant_emotion']
    except Exception as e:
        detected_emotion = "error"
    return detected_emotion.lower() == 'fear', detected_emotion

# --- Alert Functions ---
def send_alert(location_url, contacts, custom_msg):
    client = Client(TWILIO_SID, TWILIO_TOKEN)
    results = []
    for c in contacts:
        try:
            msg = client.messages.create(
                body=f"{custom_msg}\nLocation: {location_url}",
                from_=TWILIO_WHATSAPP_FROM,
                to=c
            )
            results.append(f"✅ Alert sent to {c}: {msg.sid}")
        except Exception as e:
            results.append(f"❌ Error for {c}: {e}")
    return results

def play_help_sound():
    engine = pyttsx3.init()
    engine.setProperty('rate', 180)
    engine.setProperty('volume', 1)
    phrase = "Help me! " * 5
    engine.say(phrase)
    engine.runAndWait()

# --- Logging & Analytics ---
def log_event(event_type, location_url):
    df = pd.read_csv(LOG_FILE) if os.path.exists(LOG_FILE) else pd.DataFrame(columns=['timestamp', 'event', 'location'])
    df = pd.concat([df, pd.DataFrame([{'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"), 'event': event_type, 'location': location_url}])], ignore_index=True)
    df.to_csv(LOG_FILE, index=False)

def show_history():
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        st.subheader("Detection History")
        st.dataframe(df)
        st.bar_chart(df['event'].value_counts())
    else:
        st.info("No history yet. Try triggering some alerts!")

def show_stats():
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        st.metric("Total Alerts", len(df))
        st.metric("Hand Detections", (df['event']=='hand_detected').sum())
        st.metric("Fear Detections", (df['event']=='fear_detected').sum())
        st.metric("SOS Alerts", (df['event']=='SOS').sum())

# --- Photo Save ---
def save_photo(frame):
    filename = f"{PHOTO_DIR}/alert_photo_{int(time.time())}.jpg"
    cv2.imwrite(filename, frame)
    st.success(f"Photo saved: {filename}")
    return filename

# --- UI ---
st.set_page_config(page_title="Atma Secure", layout="wide")
st.title("🛡️ Atma Secure")
st.write("Detects hand gestures and emotions, gets GPS, sends WhatsApp alerts, logs events, and supports SOS/manual/camera triggers.")

# --- Location ---
loc = streamlit_geolocation()
if loc and loc.get('latitude') and loc.get('longitude'):
    location_url = f"https://maps.google.com/?q={loc['latitude']},{loc['longitude']}"
    st.success(f"Location: [{location_url}]({location_url})")
else:
    st.warning("Please allow location access in your browser for accurate alerts.")
    location_url = "Location unavailable"

# --- Custom Message ---
custom_msg = st.text_area("Customize your emergency message", "🚨 Emergency Alert! Need help!")

# --- Camera & Detection ---
image = st.camera_input("Take a photo (for detection)")

if image is not None:
    try:
        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        if frame is not None:
            st.image(frame, caption="Snapshot", channels="BGR")

            hand_detected = detect_open_palm(frame)
            fear_detected, detected_emotion = detect_fear(frame)

            st.write(f"**Hand Detected:** {'✅' if hand_detected else '❌'}")
            st.write(f"**Emotion Detected:** {detected_emotion}")

            if hand_detected:
                log_event("hand_detected", location_url)
            if fear_detected:
                log_event("fear_detected", location_url)

            if hand_detected or fear_detected:
                st.error("🚨 Danger detected!")
                st.info("Playing help sound...")
                play_help_sound()
                photo_file = save_photo(frame)
                log_event("AutoPhoto", location_url)
                if st.button("Send Emergency Alert to All Contacts"):
                    results = send_alert(location_url, contacts, custom_msg)
                    for r in results:
                        st.write(r)
            else:
                st.success("No danger detected in this snapshot.")
        else:
            st.warning("Could not decode image. Please try again.")
    except Exception as e:
        st.error(f"Error processing image: {e}")
else:
    st.info("Take a photo to run detection.")

# --- SOS Button ---
if st.button("🚨 SOS Emergency Alert"):
    play_help_sound()
    results = send_alert(location_url, contacts, custom_msg)
    for r in results:
        st.write(r)
    log_event("SOS", location_url)

# --- Women Helpline Section ---
st.header("📞 Women Helpline")
st.markdown(f"""
If you are in trouble and need immediate help, call the National Women Helpline (India):  
**<span style="color:red;font-size:22px;">{WOMEN_HELPLINE_NUMBER}</span>**

[Click to Call](tel:{WOMEN_HELPLINE_NUMBER})

Or dial directly on your mobile phone keypad.
""", unsafe_allow_html=True)

# --- Analytics Dashboard ---
st.header("Detection Analytics")
show_stats()
show_history()

# --- Instructions ---
st.markdown("""
---
**Instructions:**
1. Allow browser location access.
2. Take a photo using the camera widget.
3. If an open palm or 'fear' emotion is detected, send an emergency alert with your location and save a snapshot.
4. Use the SOS button for manual emergency alerts.
5. Call the Women Helpline above if you are in danger.
6. Customize your message above.
7. View detection history and analytics below.
---
""")