**KrodhApp Machine Learning Project**

This Machine Learning project is a core component of KrodhApp, an anger management application. The project involves training a Random Forest model on a comprehensive dataset and deploying the model using Flask and Render. The model is fine-tuned with personalized data provided by app users based on their anger events. The fine-tuned model then predicts the likelihood of anger and alerts users when certain thresholds are crossed.

**Project Overview :**

1. **Dataset:** The model is trained on TestData.csv with the following features:
    => Heart Rate
    => Value SpO2
    => Sleep Points
    => Stress Score
2. **Algorithm:** Random Forest
3. **Deployment:** The model is deployed on a server using Flask (app.py) and Render.
4. **Fine-Tuning:** The deployed model is fine-tuned with personalized user data.
5. **Prediction:** The fine-tuned model returns feature values with the highest probability of predicting anger, and the app alerts the user when these values are exceeded.
