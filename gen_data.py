  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		 
import numpy as np  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
# this function should return a dataset (X and Y) that will work  		  	   		 	   		  		  		    	 		 		   		 		  
# better for linear regression than decision trees  		  	   		 	   		  		  		    	 		 		   		 		  
def best_4_lin_reg(seed=1489683273):  		  	   		 	   		  		  		    	 		 		   		 		  
 
    np.random.seed(seed)
    x1 = np.random.uniform(0, 10, size=(100,))
    x2 = np.random.uniform(0, 10, size=(100,))
    y = 2 * x1 + 3 * np.sin(x2) + np.random.normal(0, 1, size=(100,))
    x = np.column_stack((x1, x2))
    return x,y
  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
def best_4_dt(seed=1489683273):  		  	   		 	   		  		  		    	 		 		   		 		  
   
    np.random.seed(seed)
    x1 = np.random.uniform(-5, 5, size=(100,))
    x2 = np.random.normal(loc=0, scale=1, size=(100,))
    y = (x1 ** 2 + np.abs(x2) + np.random.normal(0, 1, size=(100,)))
    x = np.column_stack((x1, x2))
    return x, y
  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
def author():  		  	   		 	   		  		  		    	 		 		   		 		  
   		  	   		 	   		  		  		    	 		 		   		 		  
    return "pvenieris3"  # Change this to your user ID

def study_group():
    return "pvenieris3"

if __name__ == "__main__":
    print("they call me petros")
