clear;
close all

load IrisDataAnnotated.mat;

% ( 1 )
rng(1); %set seed control
k = 3; %number of clusters
t = 1; %index for while cycle
maxiteration = 100000; %max number of iterations
    
% 3 random centroids
a = randi([1 50],1);%Uniformly distributed pseudorandom integers
b = randi([51 100],1);
c = randi([52 150],1);
centroids = [a b c];

while t < maxiteration and %error > tol
 ...
   t = t + 1;
end

final_cluster1 = find(I_assign == 1);
final_cluster2 = find(I_assign == 2);
final_cluster3 = find(I_assign == 3); 


I_error = I - I_assign;

for m = 1:150
    if  I_error(m) == -1 
        totale = sum( I_error == -1 );
        percentuale_err_totale = (totale/150)*100;
    else
    end
end



scatter3(X(1,final_cluster1),X(2,final_cluster1),X(3,final_cluster1),'red' )
hold on
scatter3(X(1,final_cluster2),X(2,final_cluster2),X(3,final_cluster2),'blue' )
hold on
scatter3(X(1,final_cluster3),X(2,final_cluster3),X(3,final_cluster3),'green' )