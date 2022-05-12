function res = Evaluate(classifier, Data, Class)

suma = zeros(1,size(Data,2));
for i = 1:size(classifier)
    suma = suma + classifier(i,4) * WeakClassifier(classifier(i,2), classifier(i,1), Data(classifier(i,3),:));
end
res = sum(sign(suma)~= Class)/size(suma,2);

end