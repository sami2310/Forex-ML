# Forex-ML
Forex price predictions using machine learning


# Results in the `Test accuracy` notebook:
1.Variables:
 - future = 15
 - averages = [5,10,15,20,30,50,70,100,200,300]
 - momentum_values = [10, 20, 30, 50, 70, 100]
 - heiken = [15]
 
After using the scaler for scikit learn

moving averages = 54%
moving averages + heikenashi = 56%
moving averages + heikenashi + momentum = 53%

2. Variables:
 - future = 15
 - averages = [2]
 - momentum_values = [3, 4, 5, 8, 9, 10]
 - heiken = [15]

moving averages = 47%
moving averages + heikenashi = 52%
moving averages + heikenashi + momentum = 51%
