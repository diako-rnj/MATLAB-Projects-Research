% visualizing raw data
load ModelReductionData
for i = 1:5
    for j = i+1:6
        figure;  % Open a new figure for each plot
        plot(X(i,:), X(j,:), 'k.', 'MarkerSize', 7);
        axis equal;
        set(gca, 'FontSize', 20);
        xlabel(['Component ', num2str(i)]);
        ylabel(['Component ', num2str(j)]);
        title(['Scatter Plot: Component ', num2str(i), ' vs ', num2str(j)]);

        % Generate a filename like "scatter_1_vs_2.png"
        filename = ['scatter_', num2str(i), '_vs_', num2str(j), '.png'];

        % Save the figure as PNG
        saveas(gcf, filename);

        % close the figure to avoid clutter
        close(gcf);
    end
end


% center the data

[~, p] = size(X);
xbar = 1/p * sum(X,2);
Xc = X - xbar * ones(1,p);

% compute SVD

[U, D, V] = svd(Xc);
sigma = diag(D); % Singular values

% --- Plot singular values ---
figure;
plot(sigma, 'bo-', 'LineWidth', 2);
xlabel('Component');
ylabel('Singular Value');
title('Singular Values of Centered Data');
grid on;
saveas(gcf, 'singular_values.png');


% Compute PCA scores: 4000 x 3
PC3 = V(:,1:3) * D(1:3,1:3);

% PC1 vs PC2
figure;
plot(PC3(:,1), PC3(:,2), 'k.', 'MarkerSize', 10);
xlabel('PC 1'); ylabel('PC 2');
title('PCA: PC1 vs PC2');
axis equal; grid on;
saveas(gcf, 'PCA_PC1_vs_PC2.png');

% PC1 vs PC3
figure;
plot(PC3(:,1), PC3(:,3), 'k.', 'MarkerSize', 10);
xlabel('PC 1'); ylabel('PC 3');
title('PCA: PC1 vs PC3');
axis equal; grid on;
saveas(gcf, 'PCA_PC1_vs_PC3.png');

% PC2 vs PC3
figure;
plot(PC3(:,2), PC3(:,3), 'k.', 'MarkerSize', 10);
xlabel('PC 2'); ylabel('PC 3');
title('PCA: PC2 vs PC3');
axis equal; grid on;
saveas(gcf, 'PCA_PC2_vs_PC3.png');





