from transformers import VitsModel, AutoTokenizer
import torch

import streamlit as st

# Set page title and icon
st.set_page_config(page_title="LiteTTS", page_icon=":speaking_head_in_silhouette:")


#@st.cache_resource
def run_russian_tts(genre, text):
    model = VitsModel.from_pretrained("joefox/tts_vits_ru_hf")
    tokenizer = AutoTokenizer.from_pretrained("joefox/tts_vits_ru_hf")

    text = text.lower()
    inputs = tokenizer(text, return_tensors="pt")
    inputs['speaker_id'] = genre

    with torch.no_grad():
        output = model(**inputs).waveform

    # scipy.io.wavfile.write("techno.wav", rate=model.config.sampling_rate, data=output[0].cpu().numpy())

    print(type(output[0].cpu().numpy()))

    return output[0].cpu().numpy(), model.config.sampling_rate


def main():
    # Header with image

    # Header
    st.title("Light App for Text-to-Speech")

    # Subtitle
    st.markdown("**This app will help you turn your text into speech using machine learning models**")

    # Some example content
    st.write("""
        Write your dialog below ! \n
        For a male voice start your line with [1]. For a female voice start your line with [2]. \n
         Eg. : \n
         [1]Привет \n
         [2]Привет
    """)

    # Creating tabs for every newspaper
    tab1, tab2 = st.tabs(["Russian TTS", "Credits & Legal"])

    with tab1:

        # st.text_input("Dialog here...", key="dialog")
        t = st.text_area("Type dialog below", height=200)

        if t is not None:
            textsplit = t.splitlines()

            for x in textsplit:
                st.write(x)
                tts = run_russian_tts(1, x)
                print(type(tts))
                st.audio(data=tts[0], format="audio/wav", start_time=0, sample_rate=tts[1], end_time=None, loop=False)

    with tab2:
        st.write("""
        ### Credits and Legal Disclaimer

        **Intellectual Property Notice:**
        The news titles displayed in this application are sourced from The Economist and The Financial Times websites. All news articles and titles are the intellectual property of their respective owners. The scraping and display of these titles are solely for demonstration and educational purposes within this application.

        **Legal Disclaimer:**
        - **Fair Use:** This application utilizes news titles from reputable sources to analyze sentiment. It operates under the principles of fair use, as outlined in copyright law.
        - **No Endorsement:** The presence of news titles from The Economist and The Financial Times does not imply any endorsement or affiliation with this application.
        - **Accuracy of Content:** While efforts are made to ensure the accuracy and reliability of the news titles displayed, we do not guarantee their completeness or correctness. Users should verify information from original sources for critical decision-making.

        **Terms of Use:**
        - **Third-party Content:** Users accessing news titles from The Economist and The Financial Times through this application are subject to the terms and conditions of those respective websites.
        - **Data Privacy:** This application does not collect or store any personal data from users. However, users should be aware of the data collection and privacy policies of The Economist and The Financial Times when accessing their content through this application.
        """)


if __name__ == "__main__":
    main()
