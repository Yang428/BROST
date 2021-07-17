
tracker_label = 'BROST';

tracker_command = generate_python_command('vot_wrapper', ...
    {'/home/wcz/Yang/BROST/pytracking/', ...
    '/media/wcz/datasets/yang/vot-toolkit/native/trax/support/python/',...
   '/home/wcz/Yang/BROST/' });

tracker_interpreter = 'python';

tracker_linkpath = {'/media/wcz/datasets/yang/vot-toolkit/native/trax/build/'};
