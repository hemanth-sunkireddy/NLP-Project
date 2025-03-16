# NLP Project

## Course Review Analysis and Recommendation System

Team Name: Semsantic Squad

Team Number: 60

## Scope of the project:
### Question Answering & Conversational System
The system will provide natural, context-aware answers to user queries and maintain the flow of conversation. It will consider previous questions and answers to provide relevant and coherent responses, rather than treating each question in isolation. This will allow users to ask follow-up questions or compare courses across multiple interactions, and the system will generate answers based on the entire conversation context.
### Course Review Analysis
Perform sentiment analysis on course reviews to assess the overall sentiment towards courses, instructors, and topics. This will help identify the positive and negative aspects of each course and instructor. Summarization techniques will be used to create concise summaries for instructors, courses, and topics based on the review data.
### Fake vs. Authentic Review Classification
Implement text classification models to distinguish between fake and authentic reviews. This will ensure that users are provided with reliable and credible reviews, contributing to a more trustworthy course recommendation system.
### Recommendation System (Collaborative & Content-Based Filtering)
The recommendation system will combine collaborative filtering and content-based filtering techniques. Collaborative filtering will suggest courses based on similarities between users or items, while content-based filtering will recommend courses based on the features of the courses and the user’s previous preferences. This hybrid approach will provide more personalized and accurate course recommendations for each user.

## Literature Review
https://www.mdpi.com/2076-3417/11/9/3986
This Research Paper is about Sentiment Analysis of the user’s feedback on Courses, Emotion Detection etc., (See keywords section in the research paper).

https://ieeexplore.ieee.org/abstract/document/9781308
Existing NLP Techniques for Course Review Analysis.

https://cs224d.stanford.edu/reports/LiuSingh.pdf
RNN based Recommendation System implementation from scratch.


### Dataset      
https://www.kaggle.com/datasets/septa97/100k-courseras-course-reviews-dataset/data

https://www.kaggle.com/datasets/imuhammad/course-reviews-on-coursera

https://www.kaggle.com/datasets/khusheekapoor/coursera-courses-dataset-2021


### Methodologies:
NLP for Question Answering: Use NLP techniques like transformers to provide context-aware, coherent responses and maintain conversation flow.
Sentiment Analysis and Emotion Detection: Apply sentiment analysis models to assess the sentiment of course reviews, identifying positive and negative feedback.
Text Summarization: Use extractive or abstractive summarization methods to generate concise summaries of courses, instructors, and topics from reviews.
Fake Review Detection by Text Classification: Implement machine learning models to classify reviews as fake or authentic based on content and metadata.
Hybrid Recommendation System: Combine collaborative filtering and content-based filtering to provide personalized course recommendations based on user preferences and course features.

### Evaluation Strategies:
Information retrieval-based evaluation metrics were widely used to assess the performance of systems developed for sentiment analysis. The metrics include the precision, recall,accuracy (Train Accuracy, Test Accuracy, Validation Accuracy)  and F1-score.  Statistic metrics such as the Kappa statistic and Pearson correlation are other metrics that can be used to measure the correlation between the output of sentiment analysis systems and data labeled as ground truth.
BLEU evaluates the quality of generated summaries by comparing n-grams in the generated text to reference summaries, with a higher score indicating better similarity in word choice and order. It can be used to assess the quality of course review summaries produced by the system.
AUC-ROC Curve for evaluating the Fake vs. Authentic Review Classification model.
