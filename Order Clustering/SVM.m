function [Yp,weight]=SVM(train,l_train,test,l_test)

        mdl =fitcsvm(train,l_train,'Standardize',true,'KernelFunction','RBF','KernelScale','auto');
         Yp = predict(mdl,test);
         weight=1;

end
