import boto3
import tensorflow as tf

s3 = boto3.client("s3")
model = None 

BUCKET_NAME = "ml-industrial-resnet-model"
KEY = "resnet50_pretrained.keras"

def load_model():
    global model
    if model is None:
        local_path = "/tmp/resnet50_pretrained.keras"
        s3.download_file(BUCKET_NAME, KEY, local_path)
        with open(local_path) as f:
            model = tf.keras.models.load_model(f)
    return model 

def lambda_handler(event, context):
    model = load_model()
    return {"status": "ok"}


