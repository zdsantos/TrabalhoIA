import numpy as np

class Methods:
    tol = 0.0001;
    max_iter = 1000;
    
    def gradient(self, f, p, x0, a, tol = tol, max_iter = max_iter):
        print("x0:\t\t", x0);
        print("alpha:\t\t", a);
        print("tol:\t\t", tol);
        print("max_inter:\t", max_iter);
        iter = 0;
        
        old = np.array(x0);
        x_inter = [x0]
        
        while True:
            d = - p(*old);
            new = old + a*d;
            
            x_inter.append(new);
            
            #print("xi:\t", old);
            #print("d:\t", d);
            #print("xi+1:\t", new);
            #print("f(new)=\t", f(*new));
            
            iter = iter + 1;
            err = abs(f(*old) - f(*new))
            old = new;
            if iter == max_iter or err < tol:
                break;
        
        return new, x_inter, iter;
    
    def print(self):
        print("tol: ", self.tol);
        print("max_inter: ", self.max_iter)