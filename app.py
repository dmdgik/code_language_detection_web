from flask import Flask, request, redirect, render_template, url_for
from transformers import AutoTokenizer
from src.files_io_utils import read_dict_pkl, read_yaml
from src.loading_utils import load_model_torchscript, s3_download_model
from src.app_modules import InputForm
from src.model_infer import model_infer
from src.set_secrets import set_env_secrets
import os
from loguru import logger


logger.add("./app_logs.log", rotation="1 GB", retention="2 days", level="INFO")
logger.info("Launching APP...")

APP_CONFIG_PATH = "./configs/app.yaml"
SECRETS_CONFIG_PATH = "./configs/secrets.yaml"
set_env_secrets(SECRETS_CONFIG_PATH)

app = Flask(__name__)

app_config = read_yaml(APP_CONFIG_PATH)
app.config.update(
    dict(
        SECRET_KEY=os.environ["APP_SECRET_KEY"],
        WTF_CSRF_SECRET_KEY=os.environ["WTF_CSRF_SECRET_KEY"],
    )
)

languages_list_map = read_dict_pkl(app_config["LANGUAGES_LIST_MAP_PATH"])
tokenizer = AutoTokenizer.from_pretrained(app_config["tokenizer_path"])
model = load_model_torchscript(app_config["local_model_file"], device="cpu")

logger.info("APP LAUNCHED")


@app.route("/", methods=["GET", "POST"])
def hello_world():
    logger.info(f"New request: {request.method}")
    text_form = InputForm()
    if request.method == "POST":
        if text_form.validate_on_submit():
            logger.info("Form validated")
            
            src_text = text_form.body.data
            
            logger.info(f"Input text:::  {src_text}")
            
            prediction_class, language_str, execution_time = model_infer(
                src_text, model, tokenizer, languages_list_map
            )
            processed_data = [
                str(prediction_class),
                language_str,
                str(round(execution_time, 3) * 1000),
            ]
            processed_data = " ".join(processed_data)
            
            logger.info(f"Results:::  {processed_data}")
            
            return redirect(url_for("hello_world", processed_data=processed_data))

    processed_data = request.args.get("processed_data", "")
    if len(processed_data) == 0:
        prediction_class = ""
        language_str = ""
        execution_time = ""
    else:
        result_data = processed_data.split(" ")
        prediction_class = result_data[0]
        language_str = result_data[1]
        execution_time = result_data[2]

    return render_template(
        "text_field.html",
        form=text_form,
        model_result_num=prediction_class,
        model_result_str=language_str,
        model_execution_time=execution_time,
        languages_list_map=languages_list_map
    )


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")
