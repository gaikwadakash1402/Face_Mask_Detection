import streamlit as st
import os
import subprocess


def main():
    # Title and sidebar
    st.title("Face Mask Detection App")
    st.sidebar.title("Menu")

    # Sidebar options
    option = st.sidebar.selectbox("Choose an action", ["Home", "Train Model", "Check Mask Detection"])

    if option == "Home":
        # Home page
        st.write("Welcome to the Face Mask Detection App!")
        st.write("Navigate using the sidebar to train the model or perform real-time mask detection.")

    elif option == "Train Model":
        # Model training page
        st.write("### Train the Face Mask Detection Model")
        st.write("Click on the train button to train your model first!")

        if st.button("Train"):
            st.write("Training in progress... Please wait.")
            try:
                # Execute the training script
                result = subprocess.run(["python", "train_mask_detector.py"], capture_output=True, text=True)
                st.text(result.stdout)
                st.text(result.stderr)
                st.write("Model training completed!")

                # Display training results
                if os.path.exists("plot.png"):
                    st.image("plot.png", caption="Training Loss and Accuracy")
            except Exception as e:
                st.error(f"Error during training: {e}")

    elif option == "Check Mask Detection":
        # Real-time mask detection page
        st.write("### Real-Time Mask Detection")
        st.write("Click 'Check' to activate your camera and detect masks in real-time.")

        if st.button("Check"):
            st.write("The camera feed will appear in a separate window. Press 'Q' in the video window to quit.")
            try:
                # Execute the mask detection script
                subprocess.run(["python", "detect_mask_video.py"])
            except Exception as e:
                st.error(f"Error while running mask detection: {e}")


if __name__ == "__main__":
    main()
