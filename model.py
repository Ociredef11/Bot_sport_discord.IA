from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Carica il modello
model = load_model("keras_Model.h5", compile=False)

# Carica le etichette (attenzione: devono essere esattamente queste nell’ordine giusto!)
class_names = open("labels.txt", "r").readlines()

# Dizionario con descrizioni per ogni sport
sport_descriptions = {
    "Calcio": (
        "Il calcio è uno sport di squadra giocato da due squadre di undici giocatori. "
        "L'obiettivo è segnare nella porta avversaria usando principalmente i piedi. "
        "È lo sport più popolare al mondo, praticato ovunque. "
        "Richiede resistenza, tecnica, spirito di squadra e tattica. "
        "Il campo è grande e il ritmo può essere molto intenso. "
        "Esistono anche varianti come il calcetto o il calcio a 5. "
        "È spesso considerato molto più di uno sport: una vera passione globale."
    ),
    "Basket": (
        "Il basket è uno sport dinamico dove due squadre da cinque giocatori si sfidano a segnare nel canestro avversario. "
        "È molto veloce, ricco di azioni spettacolari come schiacciate e tiri da tre punti. "
        "Serve coordinazione, agilità, visione di gioco e precisione. "
        "Ogni partita è divisa in quarti e si gioca in un palazzetto. "
        "È praticato a livello globale, in particolare negli USA con l’NBA. "
        "Il gioco è nato nel 1891 e da allora è diventato uno sport olimpico. "
        "Molto amato anche per la sua cultura street e urbana."
    ),
    "Hockey su pista": (
        "L'hockey su pista è uno sport di squadra giocato su una superficie liscia con pattini tradizionali (non in linea). "
        "Si gioca con una pallina e due squadre da cinque giocatori (compreso il portiere). "
        "Serve velocità, controllo del bastone, e ottimo gioco di squadra. "
        "È praticato soprattutto in Europa, in paesi come Italia, Spagna e Portogallo. "
        "Le partite sono intense e spettacolari, con tanti tiri e azioni veloci. "
        "È diverso dall’hockey su ghiaccio perché non si gioca sul ghiaccio e non si usano pattini a lama. "
        "Lo sport è regolato dalla World Skate Federation."
    ),
    "Snowboard": (
        "Lo snowboard è uno sport da neve in cui si scivola su una tavola su piste innevate. "
        "È nato negli anni '60 negli Stati Uniti e unisce elementi di surf, skate e sci. "
        "Esistono diverse discipline: freestyle, slalom, half-pipe, cross e freeride. "
        "Richiede equilibrio, forza, controllo e creatività nei movimenti. "
        "È sport olimpico dal 1998 e molto popolare tra i giovani. "
        "Può essere praticato sia per competizione che per puro divertimento. "
        "Shaun White è uno degli atleti più famosi di sempre in questo sport."
    )
}

def get_class(image_path):
    # Prepara immagine
    image = Image.open(image_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    description = sport_descriptions.get(class_name, " Sport non riconosciuto. Forse non è tra quelli nel modello.")

    return f" Sport riconosciuto: **{class_name}**\n Confidenza: {confidence_score:.2f}\n\n{description}"
