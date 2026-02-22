function [Yp,weight]=DT(train,l_train,test)
    tree =ClassificationTree.fit(train,l_train);
%     view(tree,'mode','graph')
    Yp = predict(tree,test);
    weight=1.5;
    
end