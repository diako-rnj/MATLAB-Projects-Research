% ex1.

% Load data
load IrisDataAnnotated.mat  
X = X';                     
I_true = I(:);             

k = 3;
num_runs = 3;

for run = 1:num_runs
    fprintf('\n=== Run %d ===\n', run);
    rng('shuffle'); 

    % K-means
    [idx_kmeans, ~] = my_kmeans(X, k, 1e-4);
    fprintf('K-means Misclassifications:\n');
    report_misclassifications(I_true, idx_kmeans, k);

    % K-medoids 
    [idx_kmedoids, ~] = my_kmedoids(k, X, 1e-4);
    fprintf('K-medoids Misclassifications (L1 distance):\n');
    report_misclassifications(I_true, idx_kmedoids, k);
end


function report_misclassifications(true_labels, cluster_labels, k)
confusion = zeros(k, k);
for i = 1:k
    for j = 1:k
        confusion(i,j) = sum((true_labels == i) & (cluster_labels == j));
    end
end


[~, cluster_to_class] = max(confusion);

mapped_pred = zeros(size(cluster_labels));
for j = 1:k
    mapped_pred(cluster_labels == j) = cluster_to_class(j);
end


total_errors = sum(mapped_pred ~= true_labels);
fprintf('Total misclassifications: %d\n', total_errors);
for i = 1:k
    errors = sum((true_labels == i) & (mapped_pred ~= i));
    fprintf('  Species %d: %d misclassified\n', i, errors);
end
end

% K-means Algorithm 
function [I, C] = my_kmeans(X, k, tau)
[n, d] = size(X);
rng('shuffle');
perm = randperm(n);
C = X(perm(1:k), :); 

D = compute_distance_matrix(X, C);
[~, I] = min(D, [], 2);

Q_old = inf;
while true
    for j = 1:k
        C(j, :) = mean(X(I == j, :), 1);
    end
    D = compute_distance_matrix(X, C);
    [q, I] = min(D, [], 2);
    Q_new = sum(q.^2);
    if abs(Q_old - Q_new) < tau
        break;
    end
    Q_old = Q_new;
end
end

function D = compute_distance_matrix(X, C)
n = size(X, 1);
k = size(C, 1);
D = zeros(n, k);
for i = 1:n
    for j = 1:k
        D(i,j) = sqrt(sum((X(i,:) - C(j,:)).^2));
    end
end
end

% K-medoids Algorithm with L1 Distance
function [I, I_bar] = my_kmedoids(k, X, tau)
n = size(X, 1);
rng('shuffle');

ninit = 20;
Q_best = inf;

for t = 1:ninit
    I_try = randperm(n, k);
    Dm = compute_L1_matrix(X, X(I_try, :));
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
    Dm = compute_L1_matrix(X, X(I_bar, :));
    [q, I] = min(Dm, [], 2);
    Q_new = sum(q);
    if abs(Q_old - Q_new) < tau
        break;
    end
    Q_old = Q_new;
    for j = 1:k
        idx = find(I == j);
        if isempty(idx), continue; end
        D_sub = compute_L1_matrix(X(idx,:), X(idx,:));
        [~, ind] = min(sum(D_sub, 2));
        I_bar(j) = idx(ind);
    end
end
end

function D = compute_L1_matrix(X, C)
n = size(X, 1);
k = size(C, 1);
D = zeros(n, k);
for i = 1:n
    for j = 1:k
        D(i,j) = sum(abs(X(i,:) - C(j,:)));
    end
end
end



% ex2.

load BiopsyData.mat  

X = X(:, 1:length(I))';  
I = I(:);            

nan_rows = any(isnan(X), 2);
X_clean = X(~nan_rows, :);    
I_clean = I(~nan_rows);      

fprintf('Samples after cleaning:\n');
fprintf('  Malignant: %d\n', sum(I_clean == 1));
fprintf('  Benign   : %d\n\n', sum(I_clean == 0));

k = 2;
num_runs = 3;

for run = 1:num_runs
    fprintf('=== Run %d ===\n', run);
    rng('shuffle');

    [idx, ~] = my_kmedoids(k, X_clean, 1e-4);

    confusion = zeros(k, k);
    for i = 1:k
        for j = 1:k
            confusion(i,j) = sum((I_clean == (i-1)) & (idx == j));
        end
    end

    [~, cluster_to_class] = max(confusion);

    mapped_pred = zeros(size(idx));
    for j = 1:k
        mapped_pred(idx == j) = cluster_to_class(j) - 1;
    end

   
    total_errors = sum(mapped_pred ~= I_clean);
    fprintf('Total misclassifications: %d out of %d\n', total_errors, length(I_clean));

    true_malignant = (I_clean == 1);
    true_benign = (I_clean == 0);
    predicted_malignant = (mapped_pred == 1);
    predicted_benign = (mapped_pred == 0);

    TP = sum(true_malignant & predicted_malignant);
    TN = sum(true_benign & predicted_benign);
    FN = sum(true_malignant & predicted_benign);
    FP = sum(true_benign & predicted_malignant);

    if TP + FN > 0
        sensitivity = TP / (TP + FN);
    else
        sensitivity = NaN;
    end

    if TN + FP > 0
        specificity = TN / (TN + FP);
    else
        specificity = NaN;
    end

    fprintf('Sensitivity (Recall for malignant): %.2f%%\n', 100 * sensitivity);
    fprintf('Specificity (True negative rate): %.2f%%\n\n', 100 * specificity);
end



% ex3.

load CongressionalVoteData.mat  

X = X'; 
I = I(:);  
n = size(X, 1);
D = zeros(n);

for i = 1:n
    for j = 1:n
        valid = (X(i,:) ~= 0) & (X(j,:) ~= 0); 
        if sum(valid) == 0
            D(i,j) = 0.5;
        else
            D(i,j) = sum(X(i,valid) ~= X(j,valid)) / sum(valid);
        end
    end
end

% k-medoids
k = 2;
[idx, medoid_indices] = my_kmedoids(k, D, 1e-4);

C = zeros(2,2);
C(1,1) = sum((I == 1) & (idx == 1));
C(1,2) = sum((I == 1) & (idx == 2));
C(2,1) = sum((I == 0) & (idx == 1));
C(2,2) = sum((I == 0) & (idx == 2));

disp('Clustering vs. Party Matrix C:');
disp(array2table(C, ...
    'VariableNames', {'Cluster_1', 'Cluster_2'}, ...
    'RowNames', {'Democrats (1)', 'Republicans (0)'}));

disp('Medoid voting patterns (1 = yes, -1 = no, 0 = missing):');
for m = 1:k
    fprintf('Medoid %d:\n', m);
    disp(X(medoid_indices(m), :));
end

% PCA Visualization
X_centered = X - mean(X, 1, 'omitnan');
[~, ~, V] = svd(X_centered, 'econ');
score = X_centered * V;

% Plot PCA of clustered points
figure;
gscatter(score(:,1), score(:,2), idx);
xlabel('1st Principal Component');
ylabel('2nd Principal Component');
title('PCA of Congressional Votes (colored by k-medoids cluster)');
set(gca, 'FontSize', 12);
saveas(gcf, 'congress_pca_clusters.png');