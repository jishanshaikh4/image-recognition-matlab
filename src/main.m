clear;

camera_type = webcam;
neural_nework = alexnet;

while(True):    % Taking Realtime images continuously.
  picture_name = camera_type.snapshot;
  picture_name = imresize(picture_name, [227, 227]);   % Resizing image to [227, 227] because alexnet is trained for such size specifically.
  new_label = classify(neural_network, picture_name);
  image(picture_name);    % Display the image taken by webcam.
  title(char(new_label));   % Assigning the title to the image taken by webcam at runtime (Realtime).
  drawnow;
end
