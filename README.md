# textCategoryPrediction
This is the start of a file that takes a list of text and categories and uses that data to predict new text. Right now there are only four different models that the code checks. It selects the model with the best testing accuracy and uses that on the real data. I wll add more models and eventually allow the user to choose what models they would like to use but, until then, the four models should suffice.

Instructions
1. Provide the path of the csv file that contains the data you would like to train the models on. (e.g. /Users/Zach/Desktop/python/file.csv) This file is also the file that is written to everytime you predict a new string of text. Unless the text is already contained in the file in which the text will not be written to the file to prevent overfitting.
2. Type the text you would like to predict. Let's say you're testing if a subject line is spam or ham. You could type "FREE EVERYTHING" without the quotes
3. The next question is a weird csv file thing. If you're using a mac and have not used the csv library in python to write the file, the first time the file is written to, it will append the new text to the last line rather than skipping to the next line. After writing to the file once, it won't happen again but it's just the first time. So if you're on a mac and have not written to the file using the python csv file then type "yes" without the quotations otherwise type "no" without the quotations.
4. Next, the program will provide the predicted category. Type "yes" without the quotation marks if it is correct and type "no" without the quotation marks if it is incorrect
5. If you typed no, it will prompt you to enter the correct category.

That's it! Predict away!
