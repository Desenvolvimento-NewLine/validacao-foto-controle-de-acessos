from distutils.core import setup
import py2exe

setup(
    windows=['snapAssistant.py'],
    options={
        'py2exe': 
        {
            'includes': ['sip', 'PyQt5.QtWidgets', 'cv2', 'numpy'],
        }
    }
)
