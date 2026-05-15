%%%%%%%%%%%%%%
rng('default'); % For reproducibility
X = [randn(100,2)*0.75+ones(100,2);
    randn(100,2)*0.55-ones(100,2)];
figure;
plot(X(:,1),X(:,2),'.');
title('Randomly Generated Data');

%WARNING: N=number data ; P= features data
% kmedoids partitions the points in the N-by-P data matrix X into K clusters 
%returns the K cluster medoid locations in the K-by-P matrix C.
%IDX N-by-1 vector containing the cluster indices of each point.  
%By default, kmedoids uses squared Euclidean distances.
    
opts = statset('Display','iter');
[idx,C] = kmedoids(X,2,'Algorithm','pam','Options',opts); 

figure;
plot(X(idx==1,1),X(idx==1,2),'r.','MarkerSize',7)
hold on
plot(X(idx==2,1),X(idx==2,2),'b.','MarkerSize',7)
plot(C(:,1),C(:,2),'co',...
     'MarkerSize',7,'LineWidth',1.5)
legend('Cluster 1','Cluster 2','Medoids',...
       'Location','NW'); %opzione per posizionare la legend
title('Cluster Assignments and Medoids');
hold off
