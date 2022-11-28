import streamlit as st
import pandas as pd
import pickle
import time
import random
import matplotlib.pyplot as plt
import seaborn as sns
import keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import librosa
import numpy as np
from librosa import stft


class audioFile:

    def __init__(self, filename, normalize=False, root_path=""):

        self.audioSignal, self.samplingFrequency = librosa.load(
            path=filename, sr=None)

        if normalize:
            average = np.mean(self.audioSignal)
            std_deviation = np.std(self.audioSignal)
            self.audioSignal = (self.audioSignal - average)/std_deviation

        self.length = len(self.audioSignal)

    def spectrogram(self, dt=0.025):

        return np.abs(stft(self.audioSignal, n_fft=int(self.samplingFrequency * dt),
                           hop_length=int(self.samplingFrequency * dt)))

    def logMelSpectrogram(self, dt=0.025):

        spectrogram = self.spectrogram(dt)
        num_spectrograms_bins = spectrogram.T.shape[-1]

        linear_to_mel_weight_matrix = librosa.filters.mel(
            sr=self.samplingFrequency,
            n_fft=int(dt*self.samplingFrequency) + 1,
            n_mels=num_spectrograms_bins).T

        mel_spectrogram = np.tensordot(
            spectrogram.T,
            linear_to_mel_weight_matrix,
            1)

        return np.log(mel_spectrogram + 1e-6)

    def plotSpectrogram(self, dt=0.025):

        spectrogram = self.spectrogram(dt)

        sns.heatmap(np.rot90(spectrogram.T), cmap='inferno',
                    vmin=0, vmax=np.max(spectrogram)/3)
        loc, labels = plt.xticks()
        l = np.round((loc-loc.min())*self.length /
                     self.samplingFrequency/loc.max(), 2)
        plt.xticks(loc, l)
        loc, labels = plt.yticks()
        l = np.array(loc[::-1]*self.samplingFrequency/2/loc.max(), dtype=int)
        plt.yticks(loc, l)
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")

    def plotLogMelSpectrogram(self, dt=0.025):

        logMelSpectrogram = self.logMelSpectrogram(dt)
        sns.heatmap(np.rot90(logMelSpectrogram),
                    cmap='inferno', vmin=-6)
        loc, labels = plt.xticks()
        l = np.round((loc-loc.min())*self.length /
                     self.samplingFrequency/loc.max(), 2)
        plt.xticks(loc, l)
        plt.yticks([])
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Mel)")

    def normalizeLength(self, duration=1):

        normalizedSamples = duration * self.samplingFrequency

        if self.length >= normalizedSamples:
            self.audioSignal = self.audioSignal[:int(normalizedSamples)]
            self.length = normalizedSamples

        else:
            self.audioSignal = np.concatenate(
                [self.audioSignal, np.zeros(int(normalizedSamples - self.length))])
            self.length = normalizedSamples
        return self

    def addWhiteNoise(self, amplitude=0.05):

        whiteNoise = np.random.normal(
            0, amplitude*np.max(np.abs(self.audioSignal)), self.length)
        self.audioSignal = np.array(self.audioSignal + whiteNoise)

        return self

    def play(self):

        Audio(data=self.audioSignal, rate=self.samplingFrequency)

        return

    def record(self, root_path='', sec=3):

        display(Javascript(RECORD))
        s = output.eval_js('record(%d)' % (sec*1000))
        b = b64decode(s.split(',')[1])

        filepath = root_path + self.fileName

        with open(filepath, 'wb') as f:
            f.write(b)

        return self

    def timeShifting(self, shift=2000):

        self.audioSignal = np.roll(self.audioSignal, shift)

        return self

    def subsample(self, factor):

        signal = [self.audioSignal[factor*k]
                  for k in np.arange(0, int(self.length/factor))]

        self.audioSignal = np.array(signal)
        self.length = len(signal)
        self.samplingFrequency = int(self.samplingFrequency/factor)

        return self


def main():
    st.set_page_config(layout="wide",
                       page_icon="🎤")

    pages1 = {
        "Reconnaissance vocale": page_reco_vocal,
        "Preprocessing et visualisation des données": page_preprocessing,
        "Augmentation des données": page_augmentation
    }

    pages2 = {
        "Jeu de données": page_jeu_données,
        "Modèle et résultat": page_modèle,
        "Isolation d'un locuteur": page_isolation_speaker,
        "Résultat avec isolation": page_resultat_isolation,
        "Démo reconnaissance de mot": page_demo_mot
    }

    pages3 = {
        "CTC loss": page_ctc_loss,
        "Exploration du dataset": page_explo_data,
        "Modèle": page_modèle2,
        "Pipeline de données": page_pipeline,
        "Résultats": page_résultats2,
        "Conclusion et perspectives": page_conclusion
    }

    st.sidebar.title("Reconaissance Vocale")
    choix = st.sidebar.selectbox(
        "", ("Introduction", "Classification", "Transcription"))

    if choix == "Introduction":
        page = st.sidebar.radio("", tuple(pages1.keys()))
        pages1[page]()

    if choix == "Classification":
        page = st.sidebar.radio("", tuple(pages2.keys()))
        pages2[page]()

    if choix == "Transcription":
        page = st.sidebar.radio("", tuple(pages3.keys()))
        pages3[page]()

    st.sidebar.info(
        "Projet DataScientist - Promotion Bootcamp Septembre 2022"
        "\n\n"
        "Participants:"
        "\n\n"
        "Quentin Blechet"
        "\n\n"
        "Nicolas Hircq"
        "\n\n"
        "Philippe Moussa"
        "\n\n"
        "Mentor:"
        "\n\n"
        "Paul Lestrat"
    )


def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        curr_frequency = List.count(i)
        if (curr_frequency > counter):
            counter = curr_frequency
            num = i

    return num


def voting_classifier(model_list, wav_file):
    predictions = []
    for model in model_list:
        predictions.append(np.argmax(model.predict(wav_file), axis=1))

    return most_frequent(predictions)


def random3():
    return [int(2500*random.random()),
            int(2500*random.random()),
            int(2500*random.random())]


def getTranscripts(df_results, ranks, epoch1, epoch2):

    header1 = 'v12big_' + epoch1
    header1_d = header1 + '_d'
    header2 = 'v12big_' + epoch2
    header2_d = header2 + '_d'

    df = pd.DataFrame()

    label1 = df_results['label'].values[ranks[0]]
    score1_1 = str(round(df_results[header1_d].values[ranks[0]], 2))
    pred1_1 = df_results[header1].values[ranks[0]]
    score1_2 = str(round(df_results[header2_d].values[ranks[0]], 2))
    pred1_2 = df_results[header2].values[ranks[0]]

    label2 = df_results['label'].values[ranks[1]]
    score2_1 = str(round(df_results[header1_d].values[ranks[1]], 2))
    pred2_1 = df_results[header1].values[ranks[1]]
    score2_2 = str(round(df_results[header2_d].values[ranks[1]], 2))
    pred2_2 = df_results[header2].values[ranks[1]]

    label3 = df_results['label'].values[ranks[2]]
    score3_1 = str(round(df_results[header1_d].values[ranks[2]], 2))
    pred3_1 = df_results[header1].values[ranks[2]]
    score3_2 = str(round(df_results[header2_d].values[ranks[2]], 2))
    pred3_2 = df_results[header2].values[ranks[2]]

    df['Sample'] = ['Audio1', 'Audio2', 'Audio3']
    df['Labels'] = [label1, label2, label3]
    df['Predictions epoch ' + epoch1] = [pred1_1, pred2_1, pred3_1]
    df['Acc. ' + epoch1] = [score1_1, score2_1, score3_1]
    df['Predictions epoch ' + epoch2] = [pred1_2, pred2_2, pred3_2]
    df['Acc. ' + epoch2] = [score1_2, score2_2, score3_2]
    df.set_index('Sample')
    return df


def randomDisplay(df_results, epoch1, epoch2):

    rank1, rank2, rank3 = random3()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write('Audio 1')
        st.audio(df_results['localPath'].values[rank1])
    with col2:
        st.write('Audio 2')
        st.audio(df_results['localPath'].values[rank2])
    with col3:
        st.write('Audio 3')
        st.audio(df_results['localPath'].values[rank3])

    st.table(getTranscripts(df_results, [rank1, rank2, rank3], epoch1, epoch2))

# Decodeur prédictions


alphabet = [chars for chars in " ABCDEFGHIJKLMNOPQRSTUVWXYZ'"]
character_encoder = keras.layers.StringLookup(
    vocabulary=alphabet, oov_token="")
character_decoder = keras.layers.StringLookup(
    vocabulary=character_encoder.get_vocabulary(), oov_token="", invert=True)


def decode_batch_predictions(pred):

    input_len = np.ones(pred.shape[0]) * pred.shape[1]

    # Greedy search : décodage le plus rapide, ne mène pas forcément au texte le plus probable
    results = keras.backend.ctc_decode(
        pred, input_length=input_len, greedy=True)[0][0]

    # on itère sur la prédiction et on récupère le texte
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(
            character_decoder(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text


def CTC_loss(y_test, y_pred):

    batch_len = tf.cast(tf.shape(y_test)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_test)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(
        y_test, y_pred, input_length, label_length)

    return loss


def predict(model, filePath):
    logMelSpectrogram = audioFile(filePath, normalize=True).subsample(3).normalizeLength(17) \
        .logMelSpectrogram(k_temp=.7, k_freq=1.5)
    logMelSpectrogram = np.array([(logMelSpectrogram)])
    return "prediction: " + decode_batch_predictions(model.predict(logMelSpectrogram))[0]


def page_reco_vocal():
    st.title("Reconnaissance vocale")
    st.header("Fonctionnement de la reconnaissance vocale")
    st.image('Images/image1.png')

    st.write("""
             - Etude de 2 modèles :
                 - Modèle de classification
                 - Modèle de transcription

             """)


def page_preprocessing():
    st.title("Preprocessing")
    st.header("Etapes d'obtention d'un spectrogramme")
    st.image('Images/image2.png')

    st.write("""
         - Transformation de la pression acoustique en signal analogique puis conversion en signal numérique
         - Décomposition du son en ses composantes fréquentielles
         - Division du son en fenêtres temporelles permettant une représentation tridimensionnelles du signal via le spectrogramme

         """)
    st.header("Spectrogramme de MEL")

    col1, col2 = st.columns(2)

    with col1:
        st.image('Images/image3.png')
        st.subheader("Spectrogramme standard")

    with col2:
        st.image('Images/image4.png')
        st.subheader(
            "Spectrogramme de MEL basé sur la perception humaine des fréquences")


def page_augmentation():
    st.title("L'augmentation des données")
    option = st.selectbox(
        'Les différents types de transformations du signal audio',
        ('Pas de transformations', 'Bruit Blanc', 'Time Shifting', 'Time Stretching', 'Pitching'))
    if option == 'Pas de transformations':
        st.audio('Audio/standard.mp3')
        st.image('Images/standard.png')
    if option == 'Bruit Blanc':
        st.audio('Audio/Bruit blanc.mp3')
        st.image('Images/Bruit blanc.png')
    if option == 'Time Shifting':
        st.audio('Audio/shifting.mp3')
        st.image('Images/shifting.png')
    if option == 'Time Stretching':
        st.audio('Audio/stretching.mp3')
        st.image('Images/stretching.png')
    if option == 'Pitching':
        st.audio('Audio/Pitching.mp3')
        st.image('Images/Pitching.png')


def page_jeu_données():
    st.title("Jeu de données")

    col1, col2, col3 = st.columns(3)

    lst = ["6 locuteurs",
           "Prononciation des chiffres de 0 à 9",
           "3000 fichiers audio",
           "langue anglaise, différents accents",
           "Source : Free Spoken Digital Dataset"]

    with col1:
        check = st.checkbox("Caractéristiques")
        if check:
            for point in lst:
                st.markdown("- " + point)

    with col2:
        df = pd.read_csv('Dataset/audio_path.csv', sep=";")

        df0 = df[df.Class == 0].head(1)
        df1 = df[df.Class == 1].head(1)
        df2 = df[df.Class == 2].head(1)
        df3 = df[df.Class == 3].head(1)
        df4 = df[df.Class == 4].head(1)
        df5 = df[df.Class == 5].head(1)
        df6 = df[df.Class == 6].head(1)
        df7 = df[df.Class == 7].head(1)
        df8 = df[df.Class == 8].head(1)
        df9 = df[df.Class == 9].head(1)

        df_example = pd.concat(
            [df0, df1, df2, df3, df4, df5, df6, df7, df7, df8, df9])

        check = st.checkbox("Extrait du premier jeu de données")
        if check:
            st.dataframe(df_example, width=220, height=425)

    with col3:
        prepa = ["Normalisation de la taille des fichiers audio",
                 "Conversion des spectrogramme en spectrogramme MEL",
                 "3 types d'augmentation : time shifting, pitch shifting et ajout de bruit blanc"]

        check = st.checkbox("Préparation des données")
        if check:
            for prep in prepa:
                st.markdown("- " + prep)


def page_modèle():
    st.title("Modèle\n")

    col1, col2 = st.columns(2)

    a = False
    b = False

    with col1:
        check1 = st.checkbox("Convolution")
        check2 = st.checkbox("Dense")
        check3 = st.checkbox("Output")

        if check1:
            st.image("Images/Conv1D.png", width=350)

        if check2:
            st.image("Images/Dense.png", width=350)

        if check3:
            st.image("Images/Proba.png", width=230)
            a = True

    with col2:
        check = st.checkbox("Résultat")
        if check:
            opt = st.selectbox(" ", ("TrainingLoss", "Training Accuracy"))
            if opt == "TrainingLoss":
                st.image("Images/TrainingLoss.png")
            if opt == "Training Accuracy":
                st.image("Images/TrainingAccuracy.png")
            b = True

    if a == True and b == True:
        check = st.checkbox("Problématique?")
        if check:
            st.header("\n")
            st.subheader(
                "Qu'en est-il de la capacité de notre modèle à prédire sur un fichier dont le locuteur est inconnu?")


def page_isolation_speaker():
    st.title("Isolation d'un locuteur")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Création des 6 jeux de données\n")
        st.image("Images/recapProcessing.png")

    with col2:
        check = st.checkbox("Jeux d'entrainement")
        if check:
            st.subheader("6 jeux d'entrainement différents\n")
            lst = ["Chaque jeu d'entrainement regroupe 5 locuteurs et en exclut 1",
                   "Chaque jeu de test est constitué des fichiers du locuteur exclut"]

            for point in lst:
                st.markdown("- " + point)

        conditions = ["normale",
                      "avec ajout de bruit blanc",
                      "avec shifting",
                      "avec pitching",
                      "avec les 3 précédentes conditions réunies (bruit blanc + shift + pitch)"]

        check = st.checkbox("Conditions")
        if check:
            st.subheader("6 modèles entrainés dans 5 conditions\n")
            for cond in conditions:
                st.markdown("- " + cond)


def page_resultat_isolation():
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Résultats\n")
        st.write("Accuracy:")
        st.write("- Modèle normal: 0.67")
        st.write("- Modèle bruité: 0.68")
        st.write("- Modèle shift: 0.80")
        st.write("- Modèle pitch: 0.75")
        st.write("- Modèle bruité + shift + pitch: 0.65")

    with col4:
        st.header("\n")
        st.subheader("Meilleur résultat : shifting\n")
        st.image("Images/ValidationAccuracyShift.png")


def page_demo_mot():
    st.title("Prédiction d'un mot (chiffre 0-9)")
    st.header("\n")
    st.header(
        "Démo de la prédiction d'un chiffre prononcé dans un fichier audio")
    st.header("\n")

    if st.checkbox("Votes"):
        st.image("Images/votes2.png")
    else:
        st.image("Images/votes1.png")

    file_dic = {"Chiffre 0": "Audio/0_Quentin.wav", "Chiffre 1": "Audio/1_Quentin.wav",
                "Chiffre 2": "Audio/2_Quentin.wav", "Chiffre 3": "Audio/3_Quentin.wav",
                "Chiffre 4": "Audio/4_Quentin.wav", "Chiffre 5": "Audio/5_Quentin.wav",
                "Chiffre 6": "Audio/6_Quentin.wav", "Chiffre 7": "Audio/7_Quentin.wav",
                "Chiffre 8": "Audio/8_Quentin.wav", "Chiffre 9": "Audio/8_Quentin.wav"}

    option = st.selectbox("Sélectionnez un fichier audio à prédire",
                          ("Chiffre 0", "Chiffre 1", "Chiffre 2", "Chiffre 3",
                           "Chiffre 4", "Chiffre 5", "Chiffre 6", "Chiffre 7",
                           "Chiffre 8", "Chiffre 9"))

    list_model = ["model_v10train_1.h5", "model_v10train_2.h5", "model_v10train_3.h5",
                  "model_v10train_4.h5", "model_v10train_5.h5", "model_v10train_6.h5"]

    models = []
    for model in list_model:
        models.append(keras.models.load_model(
            "models_shift/" + model))

    wav_file = []

    if option:
        wav_file.append(file_dic[option])

        audio_file = open(wav_file[0], 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes)

        spectro_wav_file = [audioFile(filename).normalizeLength(
            2).logMelSpectrogram() for filename in wav_file]

        spectro_wav_file = np.asarray(spectro_wav_file)
        pred = voting_classifier(models, spectro_wav_file)

        if st.button("Réalisez la prédiction"):
            st.write("Résultat:\n")
            st.write(pred[0])


def page_ctc_loss():
    
    st.title("CTC loss")
    
    col1, col2 = st.columns([3, 6])
    
    with col1:
        st.markdown('#')
        st.markdown('#')
        st.markdown('#')
        st.write("- Plutôt que de prédire 'le' bon alignement...")
        st.markdown('#')
        st.markdown('#')
        st.markdown('#')
        st.markdown('#')
        CTC = st.checkbox("CTC: principe")     
        if CTC:
            st.write("""
                     - ...Assigner une probabilité à chaque caractère par time-step
                     - Maximiser la probabilité des alignements compatibles avec la transcription
                     - Une contrainte: longueur de prédiction inférieure ou égale au nombre de time-steps
                     """)        
        
     with col2:
        st.image('Images/spectro.jpg', width=400)
        cats = ['Images/script1.jpg', 'Images/sript2.jpg', 'Images/script3.jpg', 'Images/script4.jpg', 'Images/script5.jpg]
        placeholder = st.empty()
        k = 0
        while (not CTC):
            rank = int(k % 5)
            placeholder.image(cats[rank], width=400)
            k += 1
            time.sleep(1)
        if CTC:
            st.image('Images/scripts.jpg', width=400)


def page_explo_data():
    st.title('Exploration du dataset')
    st.header('Dataset LibriSpeech - langue anglaise')

    st.write("""
             Nous disposons de plus de 30 000 enregistrements audio (7 Go / 100h) en langue anglaise
             et de leurs transcriptions. Il nous faut :

            - normaliser la durée des enregistrements

            - maintenir la compatibilité entre la longueur des étiquettes et la résolution temporelle utilisée
             """)

    with open('Transcription/train_metadata_full', 'rb') as f:
        df_train = pickle.load(f)
    with open('Transcription/test_metadata_full', 'rb') as f:
        df_test = pickle.load(f)

    rootPathTrain = '/content/drive/My Drive/LibriSpeech/train/'
    rootPathTest = '/content/drive/My Drive/LibriSpeech/test/'

    df_full = pd.concat([df_train, df_test], axis=0)

    col1, col2 = st.columns([4, 1])

    with col1:
        maxDuration = st.slider("duration (s)", min_value=0.0, max_value=25.0,
                                step=.1)
        timeStep = st.slider("time step (ms)", min_value=0.0, max_value=30.0, step=.5,
                             value=30.0)
    with col2:
        fig = plt.figure(figsize=(15, 10))
        plt.scatter(df_full[df_full['rootPath'] == rootPathTrain]['audio duration'],
                df_full[df_full['rootPath'] == rootPathTrain]['label length'], c='orange', label='train files')
        plt.scatter(df_full[df_full['rootPath'] == rootPathTest]['audio duration'],
                df_full[df_full['rootPath'] == rootPathTest]['label length'], c='green', label='test files')
        plt.legend(fontsize=12)
        plt.xlim(left=0, right=35)
        plt.ylim(bottom=0, top=530)

        plt.vlines(x=maxDuration, ymin=0, ymax=530, colors='black')
        plt.hlines(y=maxDuration/(.001*timeStep)/2,
               xmin=0, xmax=35, colors='black')
        plt.xlabel('audio duration (s)', fontsize=12)
        plt.ylabel('label length (nb char)', fontsize=12)
        plt.title('Label length vs audio duration', fontsize=16)

        plt.xlabel('audio duration (s)', fontsize=12)
        plt.ylabel('label length (nb char)', fontsize=12)
        st.pyplot(fig)


def page_modèle2():
    st.title('Modèle')
    st.subheader("Structure dérivée de Deep Speech 2")

    col1, col2, col3 = st.columns(3)

    with col1:
        couchesConvolution = st.checkbox('Couches convolution')
    with col2:
        couchesRNN = st.checkbox('Couches RNN')
    with col3:
        output = st.checkbox('output')

    st.image('Images/model0.jpg')

    placeholder1 = st.empty()
    placeholder2 = st.empty()
    placeholder3 = st.empty()

    if couchesConvolution:
        st.image('Images/model1.jpg')
    if couchesRNN:
        st.image('Images/model2.jpg')
    if output:
        st.image('Images/model3.jpg')


def page_pipeline():
    st.title('Pipeline de données')
    st.write("\n")

    col1, col2, col3 = st.columns(3, gap="small")

    with col1:
        preprocessing = st.checkbox('Preprocessing')
    with col2:
        augmentation = st.checkbox('Augmentation')
    with col3:
        modele = st.checkbox('Modèle')

    st.write("\n")

    st.image('Images/pipeline0.jpg')

    placeholder1 = st.empty()
    placeholder2 = st.empty()
    placeholder3 = st.empty()
    placeholder4 = st.empty()

    if preprocessing:
        placeholder1.image('Images/pipeline1.jpg')
    if augmentation:
        placeholder2.image('Images/pipeline2.jpg')
    if modele:
        placeholder3.image('Images/pipeline3.jpg')
        placeholder4.image('Images/pipeline4.jpg')


def page_résultats2():

    with open('Transcription/bigTestResults', 'rb') as f:
        df_results = pickle.load(f)

    df_results['localPath'] = 'test/' + \
        df_results['fileDirectory'] + df_results['fileName']

    st.title('Résultats')

    st.header('Sélection de configuration')

    with open('Transcription/history0_valLoss.pickle', 'rb') as f:
        history0 = pickle.load(f)
    with open('Transcription/history12_valLoss.pickle', 'rb') as f:
        history12 = pickle.load(f)
    history12big = [80.67, 61.40, 53.97, 49.64,
                    47.09, 45.89, 44.51, 43.65, 43.37]

    precision_v12big = [df_results['v12big_' +
                                   str(k) + '_d'].mean() for k in range(1, 10)]

    with st.expander("CTC loss sur échantillon de validation"):
           
        st.write("""A partir d'une configuration de base,
             nous avons testé plusieurs options en boucle courte:
             12 epochs, 800 train / 200 test.
             Puis le 'champion' sélectionné a été entraîné
             sur 9 epochs, 28 000 train / 2500 test. Enfin, nous avons mesuré la précision
             de ses prédictions à l'aide d'une métrique basée sur la distance de Levenshtein.
             """)
                
        col1, col2, col3, col4 = st.columns(4, gap = "small")
            with col1:    
                baseline = st.checkbox('V0 Baseline validation loss - 800 train / 200 test')
            with col2:
                v12 = st.checkbox('V12 Champion validation loss - 800 train / 200 test')
            with col3:
                v12b = st.checkbox('V12 Champion validation loss - 28 000 train / 2 500 test')
            with col4:
                v12b_p = st.checkbox('V12 Champion precision - 28 000 train / 2 500 test')

      col1, col2 = st.columns(2)
        
        with col1:
            fig = plt.figure()
            if baseline:
                plt.plot(history0, color = 'black', label = 'baseline')
            if v12:
                plt.plot(history12, color = 'grey', label = 'champion')
            if v12b:
                plt.plot(history12big, color = 'orange', label = 'champion - full training')
            plt.legend(fontsize = 10)
            plt.xlabel('epochs')
            plt.ylabel('CTC loss')
            plt.title("CTC loss on validation data", fontsize = 10)
            st.pyplot(fig)
        with col2:
            fig = plt.figure()
            if v12b_p:
                plt.plot(precision_v12big, color = 'orange', label = 'champion - full training - greedy decoding')
                plt.scatter([8], [0.874], color = 'orange', marker = '*', label ='beam search decoding')
            plt.legend(fontsize = 10)
            plt.xlabel('epochs')
            plt.ylabel('accuracy')
            plt.title('Precision on validation data', fontsize = 10)
            st.pyplot(fig)
                          
    st.header('Prédictions sur données de validation')
    st.write("""
             Et voici les prédictions de notre 'champion' après un entraînement long!
             \nNous les affichons par groupe de trois enregistrements, avec leurs labels,
             sélectionnés aléatoirement
             dans notre dataset de validation (2500 enregistrements).
             \nVous pouvez comparer les résultats obtenus à deux epochs différents.
             \nEn sélectionnant 9 et 9_2, vous pouvez aussi comparer les prédictions de l'epoch 9,
             décodées en mode "greedy' (9) ou avec un "ctc beam search" (9_2).
             """)
    epochs = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '9_2']
    selection = st.multiselect("select 2 epochs displayed", epochs,
                               default=['5', '9'], max_selections=2)
    st.button("random select", on_click=randomDisplay(
        df_results, selection[0], selection[1]))

    st.header("Autres prédictions...")

    with st.expander("Tests"):
        st.write("""Auto-test de notre modèle!""")

        custom_objects = {"CTC_loss": CTC_loss}
        with keras.utils.custom_object_scope(custom_objects):
            model5 = keras.models.load_model('Transcription/h5model.h5')

        file_dic = {"Sample 1": "samples/proud.m4a", "Sample 2": "learn.m4a",

        option = st.selectbox("Sélectionnez un fichier audio à transcrire",
                              ("Sample 1", "Sample 2"))

        wav_file = file_dic[option]

        col1, col2 = st.columns(2)
        with col1:
            if wav_file:
                st.audio(wav_file)
        with col2:
            if wav_file:
                st.button("predict", on_click=predict(
                    model5, filePath=wav_file))


def page_conclusion():
    st.title('Conclusion et perspectives')

    st.markdown("###")
    st.write("""
             Tous nos remerciements à DataScientest et plus particulièrement à 
             notre chef de cohorte Romain Godet et à notre mentor Paul Lestrat!
             """)
    st.markdown("###")

    st.header("Quelques pistes")

    st.markdown("###")

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#")
            st.write("""
            - (Beaucoup) plus de données d'entraînement
                - Deep Speech 2: plus de 11 000h audio! 
            - Autres augmentations de données:
                - stretching, pitch shifting
                - autres bruits
            - Autres configurations pre-processing/augmentation/modèle
            - Augmenter la profondeur de notre modèle
            """)
        with col2:
            st.image('Images/audioData.jpg', width=400)

    st.markdown("##")

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#")
            st.markdown("#")
            st.markdown("#")
            st.write("""
                     - Meilleur décodage des prédictions
                     - Métrique: mesure de la qualité des prédictions
                     """)
        with col2:
            st.image('Images/MLmetrics.jpg', width=400)

    st.markdown("##")

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.write("""
                     - Modèle de langage
                     """)
        with col2:
            st.image("Images/languageModel.png", width=400)

    st.markdown("##")

    with st.container():
        st.write("...")


if __name__ == '__main__':
    main()
