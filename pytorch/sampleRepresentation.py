import torch 
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils import plotly_utils as vis
import plotly.graph_objects as go

#vis a conical frustum to sample along the ray

cx = 2.0
cy = 2.0
fx = 10.0
fy = 10.0

num_samples = 3;
near_plane = 1;
far_plane = 3;

c2w = torch.eye(4)[None, :3, :];
camera = Cameras(fx=fx, fy=fy, cx=cx, cy=cy, camera_to_worlds=c2w, camera_type=CameraType.PERSPECTIVE);
ray_bundle = camera.generate_rays(camera_indices=0);

bins = torch.linspace(near_plane, far_plane, num_samples+1)[..., None];
ray_samples = ray_bundle.get_ray_samples(bin_starts=bins[:-1, :], bin_ends=bins[1:, :]);

vis_rays = vis.get_ray_bundle_lines(ray_bundle, color="teal", length=far_plane);

#fig = go.Figure(data=[vis_rays] + vis.get_frustums_mesh_list(ray_samples.frustums))
fig = go.Figure(data=[vis_rays, vis.get_frustum_points(ray_samples.frustums)]);
fig.show()

