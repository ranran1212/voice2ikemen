import streamlit as st
import librosa
import numpy as np
import speech_recognition as sr
import pykakasi
import tempfile
import openai
import requests
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from traits_and_prompts import Extraversion, Openness, Conscientiousness, Agreeableness, E_category, O_category, C_category, A_category, instruction_1, example_1, instruction_2


# OpenAI APIキーの設定
openai.api_key = st.secrets["OPENAI_API_KEY"]

# 基準値の設定
pitch_threshold = [154, 175, 214]  # ピッチの閾値
voice_speed_threshold = [5.7, 6.6, 7.1]  # 発話速度の閾値
contrast_threshold = 23.5  # スペクトラル対比の閾値
silence_thresh = -40  # 無音の閾値


def compute_features(audio_path):
    """
    音声ファイルから主要なピッチと平均スペクトラル対比を計算する

    Args:
        audio_path (str): 音声ファイルのパス
        
    Returns:
        tuple: 主要なピッチ、平均スペクトラル対比
    """
    # 音声ファイルを読み込む
    y, sr = librosa.load(audio_path, sr=None)

    # YINアルゴリズムを使用してピッチを検出
    pitches = librosa.yin(y, fmin=80, fmax=400)

    # NaN値や無限大の値を除去するためのフィルタリング
    valid_pitches = pitches[~np.isnan(pitches)]

    # 主要なピッチを計算
    main_pitch = np.mean(valid_pitches) if len(valid_pitches) > 0 else 0

    # 平均スペクトラル対比を計算
    spectral_contrasts = librosa.feature.spectral_contrast(y=y, sr=sr)
    avg_spectral_contrast = np.mean(spectral_contrasts)

    return main_pitch, avg_spectral_contrast


def categorize_audio_by_average(audio_path):
    """
    音声ファイルの平均ピッチとスペクトラル対比をもとにカテゴリ化し、画像生成に使用する色の指針を返す

    Args:
        audio_path (str): 音声ファイルのパス
        
    Returns:
        tuple: ピッチのカテゴリ、スペクトラル対比のカテゴリ、ピッチ値、スペクトラル対比値、ピッチの高低、画像生成に使用する色の指針
    """
    # 音声ファイルの特徴を計算
    pitch, spectral_contrast = compute_features(audio_path)

    # ピッチをカテゴリ化
    if pitch <= pitch_threshold[0]:
        pitch_category = 0
        pitch_height = "low pitch"
    elif pitch <= pitch_threshold[1]:
        pitch_category = 1
        pitch_height = "low pitch"
    elif pitch <= pitch_threshold[2]:
        pitch_category = 2
        pitch_height = "high pitch"
    else:
        pitch_category = 3
        pitch_height = "high pitch"

    # スペクトラル対比をカテゴリ化
    if spectral_contrast <= contrast_threshold:
        contrast_category = "husky"
    else:
        contrast_category = "clear"

    # スペクトラル対比とピッチの高低から画像生成に使用する色の指針を決定
    if contrast_category == "husky" and pitch_height == "low pitch":
        image_color_lineage = "Choose one image color from [navy, cobalt blue, Lavender, Violet ]"
    elif contrast_category == "husky" and pitch_height == "high pitch":
        image_color_lineage = "Choose one image color from [rose pink, coral pink, yellow, gold]"
    elif contrast_category == "clear" and pitch_height == "low pitch":
        image_color_lineage = "Choose one image color from [Red, Green, emerald green, , light brown]"
    else:  # contrast_category == "clear" and pitch_height == "high pitch"
        image_color_lineage = "Choose one image color from [Light Blue, turquoise blue, white, silver]"

    return pitch_category, contrast_category, pitch, spectral_contrast, pitch_height, image_color_lineage

def speaking_rate_by_audio(audio_path):
    """
    音声ファイルから発話速度を計算する

    Args:
        audio_path (str): 音声ファイルのパス
        
    Returns:
        tuple: 1秒あたりの文字数、発話速度のカテゴリ
    """
    # 音声ファイルを読み込む
    sound = AudioSegment.from_file(audio_path)
    nonsilent_ranges = []
    thresh = silence_thresh

    # 無音でない区間を検出する
    while not nonsilent_ranges:
        nonsilent_ranges = detect_nonsilent(sound, min_silence_len=100, silence_thresh=thresh)
        thresh -= 10

    # 開始時刻と終了時刻を取得する
    start_time = nonsilent_ranges[0][0]
    end_time = nonsilent_ranges[-1][1]

    # 音声の始まりと終わりを切り取る
    processed_sound = sound[start_time:end_time]

    # 音声認識オブジェクトを作成
    recognizer = sr.Recognizer()

    # 一時的なファイルを作成
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
        temp_file_path = temp_file.name
        processed_sound.export(temp_file_path, format="wav")

        # 音声ファイルを読み込む
        with sr.AudioFile(temp_file_path) as source:
            audio_data = recognizer.record(source)

    # Google Web Speech APIを使用して音声をテキストに変換
    try:
        text = recognizer.recognize_google(audio_data, language='ja-JP')
    except (sr.RequestError, sr.UnknownValueError) as e:
        raise ValueError(f"Speech recognition failed: {e}")

    # pykakasiのインスタンスを作成
    kakasi = pykakasi.kakasi()

    # テキストをひらがなに変換
    conversion_results = kakasi.convert(text)
    hiragana_text = ''.join([item['hira'] for item in conversion_results])

    # ひらがなの文字数を取得
    hiragana_length = len(hiragana_text)

    # 時間の差を秒単位で計算
    time_difference = (end_time - start_time) / 1000.0  # pydubはミリ秒単位なので1000で割る

    # 1秒あたりの文字数を計算
    characters_per_second = hiragana_length / time_difference

    # 1秒あたりの文字数に基づいて発話速度をカテゴリ化
    if characters_per_second < voice_speed_threshold[0]:
       speaking_rate = 0
    elif voice_speed_threshold[0] <= characters_per_second < voice_speed_threshold[1]:
        speaking_rate = 1
    elif voice_speed_threshold[1] <= characters_per_second < voice_speed_threshold[2]:
        speaking_rate = 2
    else:
        speaking_rate = 3

    return characters_per_second, speaking_rate

def generate_first_prompt(pitch_category, contrast_category, speaking_rate, pitch_height, image_color_lineage):
    """
    各性格特性に対応するプロンプトとカテゴリから最初のプロンプトを生成する
    
    Args:
        pitch_category (int): ピッチのカテゴリ
        contrast_category (str): スペクトラル対比のカテゴリ
        speaking_rate (int): 発話速度のカテゴリ
        pitch_height (str): ピッチの高低
        image_color_lineage (str): 画像生成に使用する色の指針
        
    Returns:
        str: 最初のプロンプト
    """
    character_keys = [E_category[speaking_rate][pitch_category], O_category[speaking_rate][pitch_category],
                      C_category[speaking_rate][pitch_category], A_category[speaking_rate][pitch_category]]
    first_prompt = (
        instruction_1
        + Extraversion[character_keys[0]] + "\n"
        + Openness[character_keys[1]] + "\n"
        + Conscientiousness[character_keys[2]] + "\n"
        + Agreeableness[character_keys[3]] + "\n"
        + "his voice is " + contrast_category + " and " + pitch_height + "\n"
        + image_color_lineage + "\n"
        + example_1
    )

    return first_prompt

def generate_second_prompt(first_prompt):
    """
    最初のプロンプトから詳細なプロンプトを生成する
    
    Args:
        first_prompt (str): 最初のプロンプト
        
    Returns:
        str: 詳細なプロンプト
    """
    # GPT-4モデルを想定して、テキスト生成リクエストを行う
    response = openai.ChatCompletion.create(
        model="gpt-4",  # このモデル名は将来的に利用可能になることを想定
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": first_prompt}
        ]
    )
    
    # 生成されたテキストを取得して整形
    generated_text = response.choices[0].message['content'].strip()
    
    # 詳細なプロンプトを形成
    second_prompt = instruction_2 + "\n" + generated_text
    
    return second_prompt

def generate_second_prompt_with_expression(second_prompt, expression):
    """
    詳細なプロンプトに表情を追加する
    
    Args:
        second_prompt (str): 詳細なプロンプト
        expression (str): 表情
        
    Returns:
        str: 表情を含む詳細なプロンプト
    """
    second_prompt_with_expression = f"{second_prompt} \nExpression: {expression} \n prompt:"
    return second_prompt_with_expression

def generate_final_prompt(second_prompt):
    """
    詳細なプロンプトから最終的な画像生成用のプロンプトを生成する
    
    Args:
        second_prompt (str): 詳細なプロンプト
        
    Returns:
        str: 最終的な画像生成用のプロンプト
    """
    # GPT-4モデルを想定してテキスト生成リクエストを行う
    response = openai.ChatCompletion.create(
        model="gpt-4",  # このモデル名は将来的に利用可能になることを想定
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": second_prompt}
        ]
    )
    
    # 生成されたテキストを取得して整形
    final_prompt = response.choices[0].message.content

    # プロンプトを展開して表示
    expander = st.expander('prompt')
    expander.write(final_prompt)
    
    return final_prompt

def generate_image(final_prompt):
    """
    最終的な画像生成用のプロンプトからOpenAI DALL-Eを使用して画像を生成し、その画像のURLを取得する
    
    Args:
        final_prompt (str): 最終的な画像生成用のプロンプト
        
    Returns:
        str: 生成された画像のURL
    """
    response = openai.Image.create(
        model="dall-e-3",
        prompt=final_prompt,
        n=1,
        quality="hd",
        size="1024x1024"
    )
    image_url = response.data[0].url  # 画像のURLを取得
    return image_url

def pipe_generate(second_prompt, expression, file_name):
    """
    詳細なプロンプトと表情から画像を生成し、表示およびダウンロードできるようにする
    
    Args:
        second_prompt (str): 詳細なプロンプト
        expression (str): 表情
        file_name (str): 音声ファイル名
    """
    second_prompt_with_expression = generate_second_prompt_with_expression(second_prompt, expression)
    final_prompt = generate_final_prompt(second_prompt_with_expression)
    image_URL = generate_image(final_prompt)
    key = expression
    
    # 生成された画像を表示
    st.image(image_URL, caption=expression)
    
    # 画像の取得、表示、およびダウンロード
    get_image_data(image_URL, key, file_name)

def get_image_data(image_url, key, file_name):
    """
    画像URLから画像データを取得し、表示とダウンロードボタンを提供する
    
    Args:
        image_url (str): 画像のURL
        key (str): キー値
        file_name (str): 音声ファイル名
    """
    # 画像データを取得
    response = requests.get(image_url)
    image_data = response.content

    # ダウンロードボタンを表示し、ユーザーが画像をダウンロードできるようにする
    st.download_button(
        label="Download Image -" + str(key),
        data=image_data,
        file_name=str(key) + file_name + ".png",
        mime="image/png",
        key=key
    )

def main():
    st.title("DIALS2 - キャストイラスト生成")
    uploaded_file = st.file_uploader("👇音声ファイルをアップロードしてください", type=["mp3"])
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name[-4:]) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
            file_name = uploaded_file.name
            
            # 音声ファイルから特徴を計算
            features = compute_features(temp_file_path)
            
            # 特徴から性格特性を推定
            pitch_category, contrast_category, pitch, contrast, pitch_height, image_color_lineage = categorize_audio_by_average(temp_file_path)
            characters_per_second, speaking_rate = speaking_rate_by_audio(temp_file_path)
            
            # 性格特性から初期プロンプトを生成
            first_prompt = generate_first_prompt(pitch_category, contrast_category, speaking_rate, pitch_height, image_color_lineage)
            
            # 初期プロンプトから詳細なプロンプトを生成
            second_prompt = generate_second_prompt(first_prompt)
            
            # 詳細プロンプトから画像を生成
            image_url = generate_image(second_prompt)
            
            # 生成された画像を表示
            st.image(image_url, caption="nomal Image")
            
            # 画像の取得、表示、およびダウンロード
            key = "nomal_image"
            get_image_data(image_url, key, file_name)
            
            # 画像生成と保存のパイプライン
            pipe_generate(second_prompt, "serious", file_name)
            pipe_generate(second_prompt, "grinning", file_name)
            pipe_generate(second_prompt, "winking", file_name)
            pipe_generate(second_prompt, "laughing", file_name)
            pipe_generate(second_prompt, "smiling", file_name)

if __name__ == "__main__":
    password = st.text_input("パスワード", type="password")
    if password == st.secrets["password"]:
        main() 
    else:
        st.error("パスワードを入力してください")