import kfp
from kfp import dsl

def create_pipeline():
    data_prep_op = dsl.ContainerOp(
        name="Data Preparation",
        image="aligikian13/data-processing:latest",
        arguments=[
            "--file_path", "/data/training.csv",
            "--updated_file_path", "/data/updated_undefined_df.csv",
            "--output_train_path", "/data/train.csv",
            "--output_val_path", "/data/val.csv"
        ],
        file_outputs={
            'train_data': '/data/train.csv',
            'val_data': '/data/val.csv'
        }
    )

    hyperparameter_tuning_op = dsl.ContainerOp(
        name="Hyperparameter Tuning",
        image="aligikian13/hyperparameter_tuning:latest",
        arguments=[
            "--repository_id", "PiGrieco/OpenSesame",
            "--train_dataset", data_prep_op.outputs['train_data'],
            "--test_dataset", data_prep_op.outputs['val_data']
        ],
        file_outputs={
            'best_params': '/app/best_trial.json'
        }
    )

    training_op = dsl.ContainerOp(
        name="Training",
        image="aligikian13/data-processing:latest",
        arguments=[
            "--params_path", hyperparameter_tuning_op.outputs['best_params'],
            "--train_data_path", data_prep_op.outputs['train_data'],
            "--eval_data_path", data_prep_op.outputs['val_data'],
            "--repository_id", "PiGrieco/OpenSesame"
        ]
    )

@dsl.pipeline(
    name='Roberta Training Pipeline',
    description='Pipeline to fine-tune the roberta model'
)
def roberta_training_pipeline():
    create_pipeline()

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(roberta_training_pipeline, 'roberta_training_pipeline.yaml')