close all
clear

%
%download Statistics and Machine learning toolbox
load fisheriris
 
X = meas(:,3:4); % meas contains data 150 x 4 

figure;
plot(X(:,1),X(:,2),'k*','MarkerSize',5);
title('Fisher''s Iris Data');
xlabel('Petal Lengths (cm)'); 
ylabel('Petal Widths (cm)');

[idx,C] = kmeans(X,3) ;
%[IDX, C] = kmeans(X, K) returns the K cluster centroid locations in
    %the K-by-P matrix C.
    %kmeans returns an N-by-1 vector IDX containing the cluster
    %indices of each point. 

    figure;
plot(X(idx==1,1),X(idx==1,2),'r.','MarkerSize',9)
hold on
plot(X(idx==2,1),X(idx==2,2),'b.','MarkerSize',9)
plot(X(idx==3,1),X(idx==3,2),'g.','MarkerSize',9)
plot(C(:,1),C(:,2),'co',...
     'MarkerSize',9,'LineWidth',1.5)
legend('Cluster 1','Cluster 2','Cluster 3','Centroids',...
       'Location','NW'); %opzione per posizionare la legend
title('Cluster Assignments and CENTROIDS');
hold off

% x1 = min(X(:,1)):0.01:max(X(:,1));
% x2 = min(X(:,2)):0.01:max(X(:,2));
% [x1G,x2G] = meshgrid(x1,x2);
% XGrid = [x1G(:),x2G(:)]; % Defines a fine grid on the plot
% 
% idx2Region = kmeans(XGrid,3);
 
%gscatter(X,Y,G) creates a scatter plot of the vectors X and Y grouped
 %by G.  Points with the same value of G are shown with the same color
 % and marker. 

% figure;
% gscatter(XGrid(:,1),XGrid(:,2),idx2Region,[0,0.75,0.75;0.75,0,0.75;0.75,0.75,0],'..');
% hold on;
% plot(X(:,1),X(:,2),'k*','MarkerSize',5);
% title 'Fisher''s Iris Data';
% xlabel 'Petal Lengths (cm)';
% ylabel 'Petal Widths (cm)'; 
% legend('Region 1','Region 2','Region 3','Data','Location','SouthEast');
% hold off;