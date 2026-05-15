clear
close all

%% TO DO : choose a dataset
load IrisDataAnnotated %or any other dataset

% X dato di input (load del file di dati)
% input x^(1), ..., x^(p), k, [2,7,9]

% doppio ciclio: D(i,i+j)=norm(X(:,i)-X(:,i+j))   
% D+D' (0 sulla diagonale)

[n,p]=size(X);
    
for i=1:p
    if (i==p)
        D(i,p)=0;
    else
        for j=i+1:p
            D(i,j)=norm(X(:,i)-X(:,j),1); %distance matrix
        end
    end
end
D=D+D';
%D(1:10,1:10)

%% TO DO
%define an initial partitioning
%% TO DO
%fix k, tol, maxit
 
t=0;

while( error > tol && k < maxit) 
    t=t+1;

%% TO DO choose the starting medoids
I_m=[2,7,9]  %I_m(1)=2; I_m(2)=7; I_m(3)=9;

D_m=D(I_m,:)

[q,I_assign]=min(D_m) 

%output
Q(t)=sum(q.^2)


%%%%%%%%%%%%%
oldI_m=I_m;
clear I_m
for ell=1:3 
    ell
I_ell = find(I_assign == ell) % Indices to points in the cluster

D_ell = D(I_ell,I_ell)

s=sum(D_ell)
[q(ell),j] = min(sum(D_ell))
 
I_m(ell) = I_ell(j)
end
newI_m=I_m
oldI_m

% COMPUTE Q and error for stopping criterion
%Q(t)=...
%error= ...
 end  

%plot required by the exercise 

%do a for cycle on k, compute WCMD
%plot to verify the best k by minimizing the WCMD or by the  PCA 

