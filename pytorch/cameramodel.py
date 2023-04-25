import torch 
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils import plotly_utils as vis


cx = 20.0  #number of pixels in the +x dimensions
cy = 10.0
fx = 20.0
fy = 20.0;

#perspective camera models
c2w = torch.eye(4)[None, :3, :]

camera = Cameras(fx=fx, fy=fy, cx=cx, cy=cy, camera_to_worlds=c2w,
                 camera_type=CameraType.PERSPECTIVE);
fig = vis.vis_camera_rays(camera);
#fig.show()


#fish eye model

cx = 10.0;
cy = 10.0;
fx = 10.0;
fy = 10.0;

c2w = torch.eye(4)[None, :3, :];

camera = Cameras(fx=fx, fy=fy, cx=cx, cy=cy, camera_to_worlds=c2w, camera_type=CameraType.FISHEYE);
fig = vis.vis_camera_rays(camera);
#fig.show()

#Equirectanguler /Sphere Camera Model

cx = 20.0;   # keep the 2:1 ratio
cy = 10.0;
fx = 20.0;
fy = 20.0;

c2w = torch.eye(4)[None, :3, :];

camera = Cameras(fx=fx, fy=fy, cx=cx, cy=cy, camera_to_worlds=c2w, camera_type=CameraType.EQUIRECTANGULAR);
fig = vis.vis_camera_rays(camera);
fig.show()

    