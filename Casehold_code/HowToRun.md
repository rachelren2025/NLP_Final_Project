# How to Run
Install required dependencies
    pip install git+https://github.com/huggingface/transformers
Use Python 3.9

## pretraining loss - set ptl=True
python multiple_choice/run_multiple_choice.py --model_name_or_path zlucia/legalbert --task_name casehold --data_dir data --output_dir output --overwrite_output_dir

## fine tuning - set ptl=False
python multiple_choice/run_multiple_choice.py --task_name casehold --model_name_or_path zlucia/legalbert --data_dir data --do_train --do_eval --do_predict --evaluation_strategy steps --max_seq_length 128 --per_device_train_batch_size 16 --learning_rate 5e-6 --num_train_epochs 3 --output_dir output --overwrite_output_dir --logging_steps 1000 --fp16