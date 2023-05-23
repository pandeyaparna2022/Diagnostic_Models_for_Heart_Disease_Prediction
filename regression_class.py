import numpy as np
import pandas as pd
import scipy.stats
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


"""Creates Logictic Regression model from a given set of data

        Attributes:
            learning_rate (float): learning rate, a float representing the size of the step
            used for updating the parameters during training
            num_iter (int): The number of iterations to run the optimization 
            algorithm during training
            fit_intercept (Boolean): A boolean indicating wheather to fit an 
            intercept term (bias) in the logistic regression model
            verbose (Boolean): A boolan indicating whether to print the progress
            during training
            # """
class LogisticRegression2:

    def __init__(self, learning_rate=0.01, num_iter=10000, verbose=False,split_data=False,test_size=0.2):
        """Initialize a new instance of Logictic Regression model

            Args:
                learning_rate (float): learning rate, a float representing the size of the
                step used for updating the parameters during training
                num_iter (int): The number of iterations to run the optimization
                algorithm during training
                verbose (Boolean): A boolan indicating whether to print the progress 
                during training
                splite_data(Boolean): A boolan to split the data into training set
                and test set or not
                test_size (float): A float indicating what percentage of data to assign
                to test set
                
            Returns:
                Instance of LogisticRegression
        """
        self.learning_rate= learning_rate
        self.num_iter = num_iter
        self.verbose = verbose
        self.split_data =split_data
        self.test_size =test_size

   
    def __binary_data(self, df,outcome):
        """Separation of the data into explanatory and outcome variables"""
        
        X=df.loc[:, df.columns != outcome]
        X.to_numpy()
        y=df.loc[:, df.columns == outcome] 
        y.to_numpy()
        return X.to_numpy(), y.to_numpy()
    
    def __normalize_features(self,X):
        """ Normalize and centralize the explanatory variables"""
        X_norm = X.copy()
        X_norm = (X_norm - np.mean(X_norm, axis=0)) / np.std(X_norm, axis=0)
        return X_norm
    
    def __add_intercept(self,X):
        """Add an additional column of ones for the beta zero intercept        
        """
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def __sigmoid(self, z):
        """Sigmoid fuction for logistic regression"""
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        """Loss function to estimate the improvement of the model in each iteration"""
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def __split_data(self, df):
        """To split the given data into training and testing set. Returns two 
        dataframes: train dataframe and test dataframe"""
       
        no_of_obs = df.shape[0]
        test_indices = np.random.choice(no_of_obs, int(no_of_obs * self.test_size), replace=False)
        train_indices = np.array(list(set(range(no_of_obs)) - set(test_indices)))
               
        df_train=df[df.index.isin(train_indices)]
        df_test=df[df.index.isin(test_indices)]
        
        return df_train, df_test
   

    def fit(self, df, outcome, *explanatory):
        """Train the Logistic Regression model for the given dataset.

        Args:
            df (dataframe): Dataframe comtaining all information to train the model.
            outcome (string): A string indicating the column name of the outcome variable.
            *explanatory(string or list of strings): One string or a list of strings or
            strings separated by comma indicating the explanatory variable. If sothing is 
            provided the model uses all the variables of the dataframe (except for the outcome
            variable) as explanatory variables

        Returns:
            LogisticRegression model: Returns an instance of the LogisticRegression 
            class with the trained weights.
        """
        if self.split_data:
            df, self.df_test = self.__split_data(df)
        
        
        # make a list of all variables in the dataframe 
        self.var = list(df.columns)
        # check if any variable was specified by the user to be used as explanatory
        # variable if so only consider those as explanatory variables
        if len(explanatory) != 0:
            self.var = list(set(explanatory).intersection(var))
        # add the oucome variable to this list   
            self.var.append(outcome)
              
        self.outcome=outcome      
        # remove all the other variables from the input dataframe
        df = df[self.var]
        # make a list of only the explanatory variables
        self.explanatory_var=list(df.columns[df.columns != outcome])
        # separate the data into explanatory and outcome data
        X, y = self.__binary_data(df, outcome)
        
        # normalize and standardize the data for explainatory variable      
        X_norm =self.__normalize_features(X)
        # add a column of ones for intercept
        X_norm = self.__add_intercept(X_norm)
        # convert outcome into a vector
        y = y.flatten()
        # initialize weights/coefficients as zeros
        theta_norm = np.zeros(X_norm.shape[1])
        # for the number of iteration specified, for every iteration calculate the 
        # calculate the weight for each explanatory variable, calculate the gradient
        # update the coefficient weight based on the gradient and the learning rate
        # calculate the loss
        for i in range(self.num_iter):
            z = np.dot(X_norm, theta_norm)
            h = self.__sigmoid(z)
            gradient = np.dot(X_norm.T, (h - y)) / y.size
            theta_norm -= self.learning_rate * gradient
            y = y
            if self.verbose:
                print(self.__loss(h, y))
                    
        
        # denormalize theta with std and scale the intercept with theta_norm*mean/std
        theta_temp = np.multiply(theta_norm, np.insert(1/np.std(X, axis=0),0,1))
        # rescale intercept with means
        theta_temp[0] = theta_temp[0] + np.sum(np.multiply(theta_temp[1:],-np.mean(X, axis=0)))
        self.theta = theta_temp
        #print(self.theta)

        X = self.__add_intercept(X)
        # make a copy of the explanatory variable and insert an empty space at index 0
        ex = self.explanatory_var.copy()
        ex.insert(0, "1")
        
        # calculate the probability of the outcome of the dataset based on the 
        # coefficients we just calculated and use it to calculate covariance matrix
        prob = self.__sigmoid(np.dot(X, self.theta))
        V = np.diagflat(np.multiply(prob, (1 - prob)))
        # calculate the covariance matrix
        covLogit = np.linalg.inv(np.dot(np.dot(X.T, V), X))
        # Calculate the standard error from the covariance matrix
        self.se = np.sqrt(np.diag(covLogit))
        print("")
        print('---------------------------------------------------------------------')
        for i in range(len(self.theta)):
            if i !=(len(self.theta)-1):
                print(f"{self.theta[i]} * {ex[i]} + ", end='')
            else:
                print(f"{self.theta[i]} * {ex[i]}")

        print('                                         ')

        #return self.theta
        

    def Summary(self,df):
        """Display the summary of the logistic regression including the test statistics"""
        if self.split_data:
            df, self.df_test = self.__split_data(df)
        df = df[self.var]
        # Residual degree of freedom
        X, y = self.__binary_data(df, self.outcome)
        
        rdof = X.shape[0] - X.shape[1]
        #   Model degree of freedom
        mdof = X.shape[1] - 1

        summary = pd.DataFrame({
            'coeff': self.theta,
            'std err': self.se,
            'z': self.theta / self.se,
            'CI [.025': self.theta - 1.96*self.se,
            '.975]': self.theta + 1.96*self.se,
            #'residual_df': [rdof for _ in range(len(self.theta))]
        })

        summary['P>|z|'] = scipy.stats.norm.sf(abs(summary.z))*2
       
        ex = self.explanatory_var.copy()
        ex.insert(0, "Constant")
        summary.index = ex
        headers=list(summary.columns)
        
        print('---------------------------------------------------------------------')

        return summary


    def predict(self, df_test , threshold=0.5):
        """Predict the output for the given data using a trained logistic regression model.

        Args:
            df (dataframe): Dataframe comtaining all information to test the model.
            outcome (string): A string indicating the column name of the outcome variable.
            threshold (float, optional): A float representing the minimum probability 
            threshold for classification.Default is 0.5.

        Returns:
            numpy.array: A numpy array containing the predicted output values for 
            the input data.     
        """
        if self.split_data:
            df_test = self.df_test
        df_test = df_test[self.var]
        X_test, y_test = self.__binary_data(df_test, self.outcome)
        
        
        X_test = self.__add_intercept(X_test)
       
        z = np.dot(X_test,  self.theta)
        
        h = self.__sigmoid(z)
        return h >= threshold
    
    
    def accuracy(self, df_test, threshold=0.5):
        """
        Calculates the accuracy of the logistic regression model on the test data.
        Parameters:
        -----------
        df_test (dataframe): Dataframe comtaining all information to test the model
        accuracy.

        threshold : float, optional (default=0.5)
            The threshold value to use for the predicted probabilities.
            All probabilities above this threshold are considered positive.

        Returns:
        --------
        float
            The accuracy of the logistic regression model on the test data.
            This is defined as the number of correct predictions divided by
            the total number of predictions.
        """
        # divide the data into outcome and explanatory variables
        if self.split_data:
            df_test = self.df_test
      
        df_test = df_test[self.var]
        X_test, y_test = self.__binary_data(df_test, self.outcome)
        
        y_pred = self.predict(df_test, threshold=threshold)
        num=0
        for i in range(len(y_pred)):
            if y_pred[i] ==  y_test[i]:
                num=num+1
            
        
        return (num/len(y_test))
    
    
    def confusion_matrix(self, df_test):
        """Compute the confusion matrix for the logistic regression model.
        Parameters:
        -----------
        X_test: array-like of shape (n_samples, n_features)
        Test data.

        y_test: array-like of shape (n_samples,)
        True labels for `X_test`.

        Returns:
        --------
        confusion_matrix: array-like of shape (n_classes, n_classes)
        Confusion matrix, where `n_classes` is the number of unique classes in `y_test`.
        The rows represent the actual classes and the columns represent the predicted classes.
        The (i, j) element of the matrix represents the number of instances where the actual class
        was i and the predicted class was j.
        """
        if self.split_data:
            df_test = self.df_test
            
        df_test = df_test[self.var]
        X_test, y_test = self.__binary_data(df_test, self.outcome)
            
        y_pred = self.predict(df_test)
        
        # create a variable classes that holds the two unique outcomes of the 
        # prediction
        classes = np.unique(y_test)
        # create an empty list
        confusion_matrix = []
        # for each possible outcom count how how many predictions wre correct
        # how many were false positives and how many were false negatives 
        for i in classes:
            idx = np.where(y_pred == i)
            test_pred = y_test[idx]
            row = []
            
            for j in classes:
                row.append(np.equal(test_pred, j).sum())
            confusion_matrix.append(row)
        
        return np.array(confusion_matrix).T
    
    
    def plot_CM(self, df_test):
        if self.split_data:
            df_test = self.df_test
        
        """Visualize the accuracy of the model using confusion matrix

        Args:
            df_test (dataframe): Dataframe comtaining all information regarding 
            test data set to test the model  accuracy.

        Returns:
            None: Produces visualization or confusion matrix.

        Examples:
            >>> # Create and train a logistic regression model
            >>> model = LogisticRegression()
            >>> model.fit(X_train, y_train)
            >>> 
            >>> # Visualize the accuracy of the model using a confusion matrix
            >>> model.vizualize_results(X_test, y_test, method='confusion_matrix')
        """
        matrix=self.confusion_matrix(df_test)
        sns.heatmap(matrix, annot=True, cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
            
       
    
    
    
    
 