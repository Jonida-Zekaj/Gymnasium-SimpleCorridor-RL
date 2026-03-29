import base64
from IPython.display import HTML, display

# Method 3: Display video in Colab
def display_video_in_colab(video_path):
    """
    Display video directly in Colab notebook
    """
    
    video = open(video_path, 'rb').read()
    video_encoded = base64.b64encode(video).decode('ascii')

    video_html = f'''
    <video width="600" controls>
        <source src="data:video/mp4;base64,{video_encoded}" type="video/mp4">
    </video>
    '''

    display(HTML(video_html))