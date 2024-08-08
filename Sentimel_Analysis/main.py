import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification
from datasets import Dataset
from turkish_lm_tuner import TrainerForClassification, EvaluatorForClassification
import wandb
import torch

# wandb giriş
wandb.login(key="bbcbfb27905a78615951bd8481eb50da658a8653")

# Veriyi yükle
data = pd.read_csv('../Emotion_Detection/Hotel_readablee.csv', encoding='utf-8')
data = data.head(5000)
df_filled = data.fillna('')
data['Num'] = data['Num'].astype(object)

output_dir = r'C:\Users\Ata Onur Ozdemir\PycharmProjects\Emotion_Detection\output'

# Çıktı dizinini oluştur
os.makedirs(output_dir, exist_ok=True)

# Tokenizer'ı başlat
model_name = "boun-tabi-LMG/TURNA"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Custom Dataset Processor sınıfı
class CustomDatasetProcessor:
    def __init__(self, tokenizer, max_input_length):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length

    def load_and_preprocess_data(self, data):
        dataset = Dataset.from_pandas(data)

        def preprocess_function(examples):
            # Pozitif ve negatif yorumları birleştir
            reviews = [str(pos) + ' ' + str(neg) for pos, neg in
                       zip(examples['Positive_Review_Tr'], examples['Negative_Review_Tr'])]

            # Tokenizer'ı kullan
            tokenized_reviews = self.tokenizer(
                reviews,
                truncation=True,
                padding='max_length',
                max_length=self.max_input_length
            )

            # Ensure labels are within the correct range (0-3)
            tokenized_reviews['labels'] = [min(int(score) // 3, 3) for score in examples['Reviewer_Score']]
            return tokenized_reviews

        tokenized_dataset = dataset.map(preprocess_function, batched=True)
        return tokenized_dataset

# Dataset işlemcisini başlat
dataset_processor = CustomDatasetProcessor(tokenizer, max_input_length=512)

# Veriyi eğitim, doğrulama ve test setlerine ayır
train_data = data.sample(frac=0.8, random_state=42)
remaining_data = data.drop(train_data.index)
validation_data = remaining_data.sample(frac=0.5, random_state=42)
test_data = remaining_data.drop(validation_data.index)

# Veri setlerini ön işle
train_dataset = dataset_processor.load_and_preprocess_data(train_data)
eval_dataset = dataset_processor.load_and_preprocess_data(validation_data)
test_dataset = dataset_processor.load_and_preprocess_data(test_data)

# Eğitim parametreleri
training_params = {
    'num_train_epochs': 6,
    'per_device_train_batch_size': 2,
    'per_device_eval_batch_size': 2,
    'output_dir': output_dir,
    'evaluation_strategy': 'epoch',
    'save_strategy': 'epoch',
    'use_cpu': False,
    'run_name': "guncel",
    'gradient_accumulation_steps': 2
}

# Optimizer parametreleri
optimizer_params = {
    'optimizer_type': 'adamw',
    'scheduler': False,
}

# Test parametreleri
test_params = {
    'per_device_eval_batch_size': 2,
    'output_dir': output_dir,
}

num_labels = 4  # Dört sınıflı sınıflandırma

# CustomTrainer sınıfını tanımla
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # Girdileri modele gönder
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Kaybı hesapla
        loss = outputs.get("loss")
        if labels is not None:
            loss = loss.mean()  # Kaybı ortalama al

        if return_outputs:
            return (loss, outputs)
        return loss

    def training_step(self, model, inputs):
        print("Training step started")
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Compute loss
        loss = self.compute_loss(model, inputs)
        print("Loss computed:", loss)

        # Perform backward pass
        self.accelerator.backward(loss)
        print("Backward pass completed")

        return loss.detach()

    def save_model_output(self, step, batch_size=100):
        # Eval dataset üzerinde tahminler yap ve kaydet
        eval_dataloader = self.get_eval_dataloader(self.eval_dataset)
        output_predictions = []

        for i, inputs in enumerate(eval_dataloader):
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.get("logits")
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                labels = inputs.get("labels").cpu().numpy()
                for pred, label in zip(preds, labels):
                    output_predictions.append((pred, label))

            # Her 100 veri için çıktı kaydet
            if (i + 1) % (batch_size // self.args.per_device_eval_batch_size) == 0:
                output_file = os.path.join(output_dir,
                                           f'model_output_step_{step}_batch_{i // (batch_size // self.args.per_device_eval_batch_size)}.csv')
                pd.DataFrame(output_predictions, columns=['Prediction', 'Label']).to_csv(output_file, index=False)
                print(f"Model outputs saved to {output_file}")
                output_predictions = []  # Tahminler sıfırlanır

# Postprocess fonksiyonu
def postprocess_fn(predictions):
    # Ensure predictions is a numpy array
    predictions = np.array(predictions)

    # Eğer tahminlerinizin sınıflandırma sınırları belirliyorsa (örneğin, 0.5'ten büyükse 1, değilse 0)
    threshold = 0.5
    processed_predictions = (predictions > threshold).astype(int)

    return processed_predictions

# TrainerForClassification'ı başlatırken CustomTrainer kullan
model_trainer = TrainerForClassification(
    model_name=model_name,
    num_labels=num_labels,
    task='classification',
    optimizer_params=optimizer_params,
    training_params=training_params,
    model_save_path=os.path.join(output_dir, "hotel_reviews_classification_model"),
    test_params=test_params,
    trainer_class=CustomTrainer,
    postprocess_fn = postprocess_fn
)

# Modeli eğit ve değerlendir
trainer, model = model_trainer.train_and_evaluate(train_dataset, eval_dataset, test_dataset)

# Eğitilen modeli kaydet
model.save_pretrained(os.path.join(output_dir, "hotel_reviews_classification_model"))

# EvaluatorForClassification ile modeli değerlendir
evaluator = EvaluatorForClassification(
    model_name=model_name,
    task='classification',
    test_params=test_params,
    num_labels=num_labels,
    postprocess_fn=lambda x: np.argmax(np.array(x), axis=-1)  # postprocess_fn fonksiyonunu burada belirtiyoruz
)
model.save_pretrained('output/model_save_path')

# Test veri seti üzerinde modeli değerlendir
results = evaluator.evaluate_model(test_dataset)

# Sonuçları DataFrame'e dönüştür
results_df = pd.DataFrame(results)

# Sonuçları yeni bir CSV dosyasına kaydet
results_df.to_csv(os.path.join(output_dir, 'evaluation_results.csv'), index=False)
print("Evaluation results saved to evaluation_results.csv.")

# Geçerli çalışma dizinini kontrol et
print("Current Working Directory:", os.getcwd())
