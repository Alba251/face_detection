# face_detection

# face_detection

## Objective
  The goal of this project was to develop a single deep learning model capable of predicting three     
  facial attributes from images, gender (binary classification), age (regression), and eye positions 
  (multivariate regression) from a dataset of 5000 images

## Libraries
    This project was developed using different libraries. To install the required packages, you can use the following command:

        pip install pandas matplotlib seaborn tensorflow scikit-learn tqdm


## Development 1
  1. Preprocess of the data
     The data has been preprocess to convert the gender input in binary (male: 1 and female:0)
     Normalization of the other variables (eye position and age) to have the same scale of data in all the variables
     Separation into train (0.7), validate (0.15) and test (0.15) for the training of the models and check that the variables are balanced
     Redimensionalization of the images to 224x224 size to have the same input as the models that will be used
     Preparation of the data as input to the model, dividing the data in groups of 32 images (batch_size = 32) and using prefetch to prepare the next lote while the model train with the previous
  2. Model
     A multitast CNN was implemented using 3 differnete arquitectures as the shared backbone. From 
     here, a global average poolong will be used to convert the information into a 1D vector, and after 3 task-specific output heads were created:
      - Gender: a dense layer with a sigmoid activation function to output the probability of being male or female
      - Age: a dense layer with a linear activation function to produce a continuous value
      - Eye position: a dense layer with a linear activation function, similar to the age output

    For the loss function, binary_crossentropy was used for the gender classification task and mean squared error (MSE) was used for both age and eye position regression tasks. The model was optimized using the Adam optimizer.

  ARCHITECTURES: The 3 architectures used where the following:
    - MobileNetV2: a lightweight and fast architecture optimized for mobile devices. It uses depthwise separable convolutions and inverted residual blocks. Ideal for low-resource environments, with good performance for simpler tasks.
    - ResNet50: a deeper and more powerful network based on residual connections, which allow for better gradient flow. It is stable and accurate, commonly used for general-purpose computer vision tasks.
    - EfficientNetB0: a modern architecture designed through neural architecture search, which scales depth, width, and resolution in a balanced way. It offers a strong trade-off between performance and parameter efficiency.

## Results 1
    With this information, the following results were obtained:
    Model Variant   | Age MAE| Eye MAE| Gender Acc | Total Loss
    MobileNetV2     | 0.297	 |  0.77  |   94.4%	   |    1.38
    EfficientNetB0	| 0.013	 |  0.02  |   41.7%	   |    0.78
    ResNet50	      | 0.007	 |  0.001 |   97.0%	   |    0.11

    Among the three architectures evaluated, ResNet50 outperformed the others across all tasks, achieving the lowest errors in both regression metrics (age and eye position) and the highest classification accuracy for gender (97.0%). These results represent a significant improvement over previous experiments with data augmentation, which did not yield better performance.
    While MobileNetV2 previously showed a better balance between tasks, especially under augmented conditions, it now falls short in both regression and overall loss. 
    Similarly, EfficientNetB0, despite achieving low regression errors, failed to generalize in the classification task, with only 41.7% gender accuracy—highlighting a poor multitask representation.
    This indicates that ResNet50 not only scales effectively in capacity but also learns robust shared features that generalize well across both classification and regression objectives.



## EXTRA ############
An additional consideration has been created, considering the possibility that only MobileNetV2 could be use, and trying to improve its initial results.

## Development 2
    Considering MobileNetV2 architecture, the idea of improving the current results was considered. For that data agmentation in training_dataset was considered. 
    The data augmentation strategy implemented focused on two controlled transformations, horizontal flip to simulates mirrored faces to increase left-right variability, adjusting according the eye position and saturation adjustment (0.7, 1.0 and 1.3)

    Variant                             | Age MAE| Eye MAE |   Gender Acc  | Total Loss
    MobileNetV2                         | 0.297	 |  0.77   |     94.4%	   |    1.38
    MobileNetV2 +  Data Augmentation  	| 0.011	 |  0.002  |     58.0%	   |    0.69

    As we can see with data augmentation, age and eye lost were reduced, however gender accuracy decreased. Then two different approaches were followed to change the model, and try to increase the gender accuracy:
    - Changes in weight_loss parameters to give more relevance to gender variable 
    - Addition of a dense layer with a relu activation function to learn features specifically relevant for gender classification and improve the model's ability to capture important patterns for that output without affecting the others.

    Variant                                         | Age MAE| Eye MAE |   Gender Acc  | Total Loss
    MobileNetV2                                     | 0.297	 |  0.77   |     94.4%	   |    1.38
    MobileNetV2 +  Data Augmentation	              | 0.011	 |  0.002  |     58.0%	   |    0.69
    MobileNetV2 +  Data Augmentation + Weight_loss	| 0.53   |  0.32   |     58.0%	   |    1.98
    MobileNetV2 +  Data Augmentation + Dense layer	| 0.04	 |  0.01   |     41.0%	   |    0.81

    Unfortunately, both approaches failed to boost gender accuracy. In fact, the loss weight adjustment led to a large increase in overall loss and worse regression errors, while the dense layer reduced gender accuracy further.

## Development 3
    With these previous results, was considered that using Data augmentation was not the best approach, then instead of considering using data augmentation, the improvement of the model structure was considered. 
    For that a mix of the previous development was considered, the model with extra dense layer in each of the output heads, to improve the model's ability to capture better important features in each variable, and an increase in the weight_loss in age and eye, as the initial model had difficulties with these two variables.

    Variant                                     | Age MAE| Eye MAE |   Gender Acc  | Total Loss
    MobileNetV2                                 | 0.297	 |  0.77   |     94.4%	   |    1.38
    MobileNetV2 +  Extra Layers +  Weight Loss	| 0.036	 |  0.09   |     95.0%	   |    0.68

## Discussion
    A robust multitask CNN was built to predict facial gender, age, and eye position from a single image using a shared MobileNetV2 encoder and task-specific branches. Among all architectures and configurations tested, the best results were achieved by tuning the output heads and loss weighting without using data augmentation, althouh the error for eye_position was still high as it is needed to consider the initial normalization of the input variables.

    In order to further improve the results, several alternative strategies could be explored:

    - Progressive fine-tuning: Training the model sequentially by focusing on one task at a time (e.g., starting with age prediction, followed by eye position and gender) may help the shared layers learn more robust and transferable features.

    - Alternative data augmentation strategies: More targeted augmentation techniques—such as light rotations, other lighting variations, or GAN-based synthetic data—could enhance model robustness without negatively impacting gender classification.











