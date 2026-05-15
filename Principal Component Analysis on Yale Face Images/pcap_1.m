%===========================================

close all
clear
%load Yale_64x64.mat
load Yale_64x64.mat
faceW = 64;
faceH = 64;
numPerLine = 11;
ShowLine = 15;
Y = zeros(faceH*ShowLine,faceW*numPerLine);
for i=0:ShowLine-1
for j=0:numPerLine-1
Y(i*faceH+1:(i+1)*faceH,j*faceW+1:(j+1)*faceW) = reshape(fea(i*numPerLine+j+1,:),[faceH,faceW]);
end
end
imagesc(Y);colormap(gray);
saveas(gcf, 'faces.png')

%===========================================

%(a) Extract in a submatrix F and plot 6 different faces for 5 different individuals as gray scale images. 
% Choose faces that you think are in some sense representative between the different groups (normal, sad, happy, surprised, sleepy, wink).
X = fea;
numPerLine = 6; % faces per person
ShowLine = 5;   % number of people

F = zeros(numPerLine * ShowLine, size(X, 2));  % 30 rows (6x5), 4096 columns
Y = zeros(faceH * ShowLine, faceW * numPerLine);  % for displaying the faces

count = 1;
for i = 0:ShowLine - 1
    for j = 0:numPerLine - 1
        index = i * 11 + j + 1; 

        % Display face in the big image
        Y(i*faceH+1:(i+1)*faceH, j*faceW+1:(j+1)*faceW) = ...
            reshape(X(index, :), [faceH, faceW]);

        % Store face in F
        F(count, :) = X(index, :);

        count = count + 1;
    end
end

figure();
imagesc(Y); colormap(gray); axis off;
title('6 Faces from 5 Individuals');
saveas(gcf, 'faces_6x5.png');

%===========================================

%b) After having computed the first singular values/vectors of F, 
% plot the singular values and comment on how fast (or slowly) they decrease. 
% Notice, however, that computing the full SVD may be very slow, so use svds, 
% increasing gradually r. It may be more informative to plot their logarithms.

% PCA Singular Value Decay – Three Separate Plots for k = 10, 20, 30

% Step 1: Center the data
mean_face = mean(F', 2);              
centering = F' - mean_face;           

% Step 2: Define k values for separate plots
k_values = [10, 20, 30];              

for i = 1:length(k_values)
    k = k_values(i);

    % Compute top-k singular values
    [~, D, ~] = svds(centering, k);
    s = diag(D);  % Singular values

    % Create figure
    figure();
    semilogy(1:k, s, '-o', 'LineWidth', 2, 'MarkerSize', 8);
    xlabel('Index');
    ylabel('Log of Singular Values');
    title(['Singular Value Decay (log scale) for k = ', num2str(k)]);
    grid on;

    % Save each plot
    filename = ['singular_values_k', num2str(k), '.png'];
    saveas(gcf, filename);
end

%==========================================

%(c) Plot the first 5 feature vectors (columns of U) as gray scale images.

mean_face = mean(F', 2);
centering = F' - mean_face;
k = 5;
[U, ~, ~] = svds(centering, k);

figure();
for i = 1:k
    subplot(1, k, i);
    imagesc(reshape(U(:, i), [64, 64]));
    colormap(gray);
    axis image off;            
    title(['PC', num2str(i)], 'FontWeight', 'bold');
end

sgtitle('First 5 feature vectors', 'FontWeight', 'bold');

% Save as PNG
set(gcf, 'Position', [100, 100, 1200, 300]);  % Wider figure with fixed height
saveas(gcf, 'eigenfaces_top5_clean.png');

%===========================================

%(d) Approximate the 5 images corresponding to the columns that you selected by a linear 
% combination of the first k = 4; 8; 15 feature vectors with coefficients their principal 
% components. Each time display the approximation and difference between it and the 
% original data in the form of a gray scale image.


mean_face = mean(F', 2);
centering = F' - mean_face;

faces_to_reconstruct = 1:5;       
ks = [4, 8, 15];                  

for k = ks
    [U_k, ~, ~] = svds(centering, k);   

    figure();
    sgtitle(['Reconstruction with k = ', num2str(k)]);

    for i = 1:length(faces_to_reconstruct)
        idx = faces_to_reconstruct(i);
        original = F(idx, :)';                      
        centered = original - mean_face;            

        coeffs = U_k' * centered;                   
        reconstruction = U_k * coeffs + mean_face;  
        diff = abs(original - reconstruction);      

        % Plot original, reconstruction, and difference
        subplot(3, length(faces_to_reconstruct), i);
        imagesc(reshape(original, 64, 64)); colormap(gray); axis off;
        if i == 1, ylabel('Original'); end
        title(['Face ', num2str(i)]);

        subplot(3, length(faces_to_reconstruct), i + length(faces_to_reconstruct));
        imagesc(reshape(reconstruction, 64, 64)); colormap(gray); axis off;
        if i == 1, ylabel('Reconstructed'); end

        subplot(3, length(faces_to_reconstruct), i + 2 * length(faces_to_reconstruct));
        imagesc(reshape(diff, 64, 64)); colormap(gray); axis off;
        if i == 1, ylabel('Difference'); end
    end
    saveas(gcf, ['reconstruction_k' num2str(k) '.png']);
end