from tensorflow.keras.models import load_model
from hepinfo.util import mutual_information_bernoulli_loss


custom_objects = {
    "mutual_information_bernoulli_loss": mutual_information_bernoulli_loss
}

model = load_model("binaryMI_qkeras.keras", custom_objects=custom_objects)