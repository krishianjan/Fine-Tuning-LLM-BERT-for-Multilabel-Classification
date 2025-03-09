# Fine-Tuning-LLM-BERT-for-Multilabel-Classification
This project focus on fine-tuning a BERT model for multilabel classification using the Reuters 21578 dataset and imdb dataset as well in task 1,2. The goal is to effectively classify news articles into multiple topics simultaneously. 

The project utilized the BERT architecture, specifically the bert-base-uncased model from
Hugging Face. Tokenization: Employed the BERT tokenizer to preprocess the text data, ensuring that the input to the
model adheres to BERT's requirements (padding, truncation, and maximum length).
Multilabel Binarization: Implemented a MultiLabelBinarizer to convert the topics of the articles into a binary format
suitable for multilabel classification.
In this task, BERT was fine-tuned for binary sentiment classification using the IMDB dataset. The model was
adjusted to classify whether a movie review is positive or negative.
Dataset: IMDB (for binary classification) Model: BERT (bert-base-uncased) Training Data: 10% of the dataset was
used for fine-tuning due to computational constraints.
Epochs: 5 Learning Rate: 2e-5. Batch Size: 4
Results:Training Loss steadily decreased over epochs. Validation Accuracy improved by over 10%, meeting the task
goal.

This task focused on fine-tuning BERT for multilabel classification using the Reuters-21578 dataset. The goal was
to classify each document into one or more categories based on the topics it covers.
Dataset: Reuters-21578 (multilabel classification) Model: BERT (bert-base-uncased) with a multilabel configuration
Training Data: 15-20% of the dataset for training and evaluation.
Epochs: 10 Learning Rate: 2e-5. Batch Size: 8
Results: Training Loss significantly decreased. demonstrating strong progress in classifying multiple categories.
F1 Micro improved from 0% to over 70% by the final epoch,
Both tasks employed the pre-trained BERT model (bert-base-uncased). In each case, the Hugging Face Trainer API
was utilized for training, with specific configurations such as learning rate, number of epochs, and batch size
tailored to the dataset. Due to computational limitations, only a portion of the dataset was used for training and
evaluation.

Optimizer: AdamW . Scheduler: Linear learning rate scheduler
Results and Evaluations
Binary Classification Results (IMDB)
Training Loss: Decreased significantly over the 5 epochs.
Validation Accuracy: Improved by 10%, achieving the goal of noticeable improvement with limited data.
Model Performance: Demonstrated a strong ability to classify positive vs. negative reviews in the IMDB dataset.
Multilabel Classification Results (Reuters-21578)
Training Loss: Decreased steadily across 10 epochs.
F1 Micro Score: Improved from 0% in the first few epochs to over 70% by the final epoch.
F1 Macro Score: Remained lower, reflecting the model's difficulty in handling rare labels, but slight improvements
were seen.

These results show significant improvement in performance across both tasks, achieving better accuracy for binary
classification and strong F1 scores for multilabel classification by the end of training.
Lesson Learned

Pre-trained Models Save Time: Fine-tuning pre-trained models like BERT significantly reduces the time and
computational resources required to train a model from scratch. It allows leveraging the powerful representations
learned by these models on vast amounts of data.
The Importance of the Right Dataset: Choosing the appropriate dataset and processing it correctly plays a crucial
role in training performance. In this project, the IMDB dataset for binary classification and the Reuters dataset for
multilabel classification helped achieve solid results by fine-tuning BERT to specific tasks.
Handling Multilabel Classification is Challenging: Multilabel classification tasks present unique challenges,
especially when some labels are rare. BERT’s performance was good on common labels but struggled with less
frequent categories, as seen with lower F1 Macro scores.
Performance Tuning is Key: Adjusting learning rate, batch size, and number of epochs allowed improvements in
both binary and multilabel classification tasks. It's important to experiment with these hyperparameters to find an
optimal setting.



** Future Scope
Scaling Up with More Data and Fine-tuning: Further improvement can be achieved by using a larger portion of the
datasets or even the entire datasets for training. With more computational resources, longer training runs or more
complex models could be explored to refine accuracy.
Multilabel Optimization: The multilabel classification task showed potential, but the model struggled with rare
labels. Future work can focus on techniques such as data augmentation, label smoothing, or weighted loss functions
to address this imbalance and improve the F1 Macro score.
Explore Other Models: BERT performed well, but newer models such as RoBERTa or DistilBERT could be
explored to see if they perform better for specific tasks. Transformer-based architectures like GPT can also be
applied to see how well they handle multilabel classification.
Transfer Learning on Domain-Specific Data: Fine-tuning the BERT model on domain-specific data (e.g., medical
text or legal documents) could lead to even better results for specific industries or fields.



** Conclusion
This project successfully demonstrated the fine-tuning of BERT models for binary and multilabel classification tasks
using the Hugging Face Trainer API. By reusing the runnable pipeline across tasks and incrementally improving
model configurations, both tasks showed measurable improvements in performance. The binary classification model
achieved over a 10% improvement in accuracy, while the multilabel classification model achieved a F1 Micro score
of over 70%
