import mlrun
from kfp import dsl 

@dsl.pipeline(
    name="Breast Cancer",
    description="gets the data, trains and deploy"
)
def pipeline(
    label_column: str = "target",
    test_size: float = 0.1,
    model_name: str = "cancer_rf_model",
    n_estimators_list: list = [10, 100, 200], 
    max_depth_list: list = [2, 5, 10]       
):
    project = mlrun.get_current_project()

    data_prep_step = project.run_function(
        function="data_prep",   
        name="fetch-data",      
        handler="fetch_data",    
        outputs=["cancer_dataset"]
    )

    train_step = project.run_function(
        function="train",        
        name="train-hyperparam", 
        handler="train_model",
        inputs={"dataset": data_prep_step.outputs["cancer_dataset"]},
        params={
            "label_column": label_column,
            "test_size": test_size,
            "model_name": model_name,
        },
        hyperparams={           
            "n_estimators": n_estimators_list,
            "max_depth": max_depth_list      
        },
        selector="max.accuracy",
        outputs=["model"]
    )
    deploy_step = project.deploy_function(
        function="serving",
        models=[{          
            "key": model_name,
            "model_path": train_step.outputs["model"]
        }],
    )
