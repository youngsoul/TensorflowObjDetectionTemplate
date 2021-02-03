import os
import subprocess
import pathlib
import platform
import logging
import model_config as mc
import tarfile
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)

base_tf2_pretrained_weights_url = "http://download.tensorflow.org/models/object_detection/tf2/20200711"

this_file_dir = os.path.dirname(os.path.abspath(__file__))



def update_model_config_file():

    config_file_path = f"{mc.CHECKPOINT_PATH}/{mc.base_pipeline_file}"

    config = config_util.get_configs_from_pipeline_file(config_file_path)
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(config_file_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)

    pipeline_config.model.ssd.num_classes = len(mc.classes)
    pipeline_config.train_config.batch_size = mc.batch_size
    pipeline_config.train_config.fine_tune_checkpoint = mc.PRETRAINED_MODEL_PATH + f'/{mc.model_name}/checkpoint/ckpt-0'
    pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
    pipeline_config.train_input_reader.label_map_path = mc.TF_ANNOTATION_PATH + '/label_map.pbtxt'
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [mc.TF_ANNOTATION_PATH + '/train.record']
    pipeline_config.eval_input_reader[0].label_map_path = mc.TF_ANNOTATION_PATH + '/label_map.pbtxt'
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [mc.TF_ANNOTATION_PATH + '/test.record']

    config_text = text_format.MessageToString(pipeline_config)
    with tf.io.gfile.GFile(config_file_path, "wb") as f:
        f.write(config_text)

    print(f"Updated config file: {config_file_path}")

if __name__ == '__main__':
    models_repo_root = mc.TF_MODEL_REPO_PATH

    abs_models_repo_path = pathlib.Path(models_repo_root).absolute()
    abs_models_repo_path.mkdir(parents=True, exist_ok=True)
    research_path = f"{abs_models_repo_path}/models/research"
    object_detection_path = f"{research_path}/object_detection"

    print(abs_models_repo_path)
    print(research_path)


    # 9 update model configuration file
    update_model_config_file()