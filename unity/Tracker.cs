using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class Tracker : MonoBehaviour
{
    string coordFilePath;
    private GameObject camera;
    private Vector3 coord;
    
    void Start()
    {
        coordFilePath = Application.dataPath + "/coordinates.csv";
        camera = GameObject.Find("Camera");
    }

    void Update()
    {
        try
        {
            coord = ReadData(coordFilePath);
        }
        catch (Exception e)
        {
            Debug.Log(e);
        }

        // convert global coordinate into headset coordinate
        this.transform.position = camear.InverseTransformPoint(coord);
    }

    Vector3 ReadData(string _filePath)
    {
        string[] _data = File.ReadAllText(_filePath).Split(' ');

        float x, y, z;
        try
        {
            x = float.Parse(_data[0]);
            y = float.Parse(_data[1]);
            z = float.Parse(_data[2]);
            return new Vector3(x, y, z);
        }
        catch (Exception e)
        {
            Debug.Log(e);
            return coord; // don't update the coordinate
        }
    }
}
