using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
public class WebCam : MonoBehaviour
{

    [SerializeField] RawImage img;
    WebCamTexture webCam;


    // Start is called before the first frame update
    void Start()
    {
        WebCamDevice[] devices = WebCamTexture.devices;
        foreach (WebCamDevice device in devices)
        {
            Debug.Log(device.name);
        }
        webCam = new WebCamTexture();
        if (!webCam.isPlaying)
        {
            webCam.Play();
        }
        img.texture = webCam;

    }

   
}
