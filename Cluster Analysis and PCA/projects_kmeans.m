% ex1.

function [I, C] = my_kmeans(X, k, tau)

[n, d] = size(X);

rng(1);
perm = randperm(n);
C = X(perm(1:k), :);

D = pdist2(X, C); 
[~, I] = min(D, [], 2); 

Q_old = inf;

while true
    for j = 1:k
        C(j, :) = mean(X(I == j, :), 1);
    end


    D = pdist2(X, C);
    [q, I] = min(D, [], 2);
    Q_new = sum(q.^2);

    if abs(Q_old - Q_new) < tau
        break;
    end

    Q_old = Q_new;
end
end



function [I, I_bar] = my_kmedoids(k, D, tau)

p = size(D, 1);
rng(1);

ninit = 20;
Q_best = inf;

for t = 1:ninit
    I_try = randperm(p, k);
    Dm = D(:, I_try);
    [q, I_assign_try] = min(Dm, [], 2);
    Q_try = sum(q);
    if Q_try < Q_best
        Q_best = Q_try;
        I_bar = I_try;
        I = I_assign_try;
    end
end

Q_old = inf;
while true
    Dm = D(:, I_bar);
    [q, I] = min(Dm, [], 2);
    Q_new = sum(q);

    if abs(Q_old - Q_new) < tau
        break;
    end
    Q_old = Q_new;

    for j = 1:k
        idx = find(I == j);
        D_sub = D(idx, idx);
        [~, ind] = min(sum(D_sub, 2));
        I_bar(j) = idx(ind);
    end
end
end

% ex2.
% Load data
load WineData.mat 
X = X';
% k_mean
k = 3;
[idx_kmeans, ~] = my_kmeans(X, k, 1e-4);

% k_medoids
D = pdist2(X, X); 
[idx_kmedoids, ~] = my_kmedoids(k, D, 1e-4);


confmat_kmeans = confusionmat(I, idx_kmeans);
confmat_kmedoids = confusionmat(I, idx_kmedoids);

disp('Confusion matrix for K-MEANS:');
disp(confmat_kmeans);

disp('Confusion matrix for K-MEDOIDS:');
disp(confmat_kmedoids);

% PCA visualization
X_centered = X - mean(X, 1);
[~, ~, V] = svd(X_centered, 'econ');
score = X_centered * V;

figure;
gscatter(score(:,1), score(:,2), idx_kmeans); 
xlabel('1st Principal Component');
ylabel('2nd Principal Component');
title('PCA of Wine Data Colored by K-Means Clusters');
set(gca, 'FontSize', 12);
saveas(gcf, 'wine_pca.png');


% ex3.

% Load data
load CardiacSPECT.mat  
X = X';  

D = pdist2(X, X, 'jaccard');  

% k-medoids
k = 2;
[idx, ~] = my_kmedoids(k, D, 1e-4); 

I = I(:);
idx = idx(:);

C = zeros(2, 2);
C(1,1) = sum(I == 1 & idx == 1);  
C(1,2) = sum(I == 1 & idx == 2);  
C(2,1) = sum(I == 0 & idx == 1);  
C(2,2) = sum(I == 0 & idx == 2); 

disp('Matrix C (actual labels vs. cluster assignments):');
disp(array2table(C, ...
    'VariableNames', {'Cluster_A', 'Cluster_B'}, ...
    'RowNames', {'Abnormal (1)', 'Normal (0)'}));

% PCA visualization 
X_centered = X - mean(X, 1);
[~, ~, V] = svd(X_centered, 'econ');
score = X_centered * V;

figure;
gscatter(score(:,1), score(:,2), idx); 
xlabel('1st Principal Component');
ylabel('2nd Principal Component');
title('PCA of CardiacSPECT Data Colored by K-Medoids Clusters');
set(gca, 'FontSize', 12);
saveas(gcf, 'cardiacspect_pca.png');


% ex4.

% Load data
load CongressionalVoteData.mat 
X = X'; 

% Remove representative with all missing votes
valid_rows = any(X ~= 0, 2);
X = X(valid_rows, :);
I = I(valid_rows);

n = size(X, 1);
D = zeros(n); 

% Compute dissimilarity
for i = 1:n
    for j = 1:n
        votes_i = X(i, :) ~= 0;
        votes_j = X(j, :) ~= 0;
        both_voted = votes_i & votes_j;

        if sum(both_voted) == 0
            D(i, j) = 0.5; 
        else
            disagreed = X(i, both_voted) ~= X(j, both_voted);
            D(i, j) = sum(disagreed) / sum(both_voted);
        end
    end
end

% k-medoids
k = 2;
[idx, medoid_indices] = my_kmedoids(k, D, 1e-4);

I = I(:); idx = idx(:); 
C = zeros(2, 2);
C(1,1) = sum((I == 1) & (idx == 1)); 
C(1,2) = sum((I == 1) & (idx == 2)); 
C(2,1) = sum((I == 0) & (idx == 1)); 
C(2,2) = sum((I == 0) & (idx == 2)); 

disp('Clustering vs. Party Matrix C:');
disp(array2table(C, ...
    'VariableNames', {'Cluster_1', 'Cluster_2'}, ...
    'RowNames', {'Democrats (1)', 'Republicans (0)'}));

disp('Medoid voting patterns (1 = yes, -1 = no, 0 = missing):');
disp('Medoid 1:');
disp(X(medoid_indices(1), :));
disp('Medoid 2:');
disp(X(medoid_indices(2), :));



% ex5. 

% Load dataset
load IrisDataAnnotated.mat 
X = X';  
I = I(:);  

k = 3;

% K-means
[idx_kmeans, ~] = my_kmeans(X, k, 1e-4);

% K-medoids
D = pdist2(X, X);  
[idx_kmedoids, ~] = my_kmedoids(k, D, 1e-4);


C_kmeans = zeros(3,3);
C_kmedoids = zeros(3,3);

for i = 1:3
    for j = 1:3
        C_kmeans(i,j) = sum((I == i) & (idx_kmeans == j));
        C_kmedoids(i,j) = sum((I == i) & (idx_kmedoids == j));
    end
end

disp('3x3 matrix: True class vs K-means clusters');
disp(array2table(C_kmeans, ...
    'VariableNames', {'Cluster1','Cluster2','Cluster3'}, ...
    'RowNames', {'Setosa','Versicolor','Virginica'}));

disp('3x3 matrix: True class vs K-medoids clusters');
disp(array2table(C_kmedoids, ...
    'VariableNames', {'Cluster1','Cluster2','Cluster3'}, ...
    'RowNames', {'Setosa','Versicolor','Virginica'}));

% PCA Visualization for K-means clusters
X_centered = X - mean(X, 1);
[~, ~, V] = svd(X_centered, 'econ');
score = X_centered * V;

figure;
gscatter(score(:,1), score(:,2), idx_kmeans);
xlabel('1st Principal Component');
ylabel('2nd Principal Component');
title('Iris PCA: K-means Cluster Labels');
set(gca, 'FontSize', 12);
saveas(gcf, 'iris_kmeans_pca.png');


% PCA Visualization for K-medoids clusters
figure;
gscatter(score(:,1), score(:,2), idx_kmedoids);
xlabel('1st Principal Component');
ylabel('2nd Principal Component');
title('Iris PCA: K-medoids Cluster Labels');
set(gca, 'FontSize', 12);
saveas(gcf, 'iris_kmedoids_pca.png');