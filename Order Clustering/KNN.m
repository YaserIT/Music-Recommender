function [Yp,weight]=KNN(train,l_train,test)
weight=1;
    for i=1:size(l_train,2)
        mdl =ClassificationKNN.fit(train,l_train(:,i));
        Yp(i,:) = predict(mdl,test);
    end
    Yp=Yp';
    
end
