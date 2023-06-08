using System;
using System.Net;
using System.Net.Sockets;
using System.Globalization;
using System.Text;
using UnityEngine;

public class UDPServer : MonoBehaviour
{
    UdpClient udpClient;
    IPEndPoint remoteEP;

    void Start()
    {
        udpClient = new UdpClient(12345);
        udpClient.BeginReceive(OnReceived, udpClient);
        remoteEP = new IPEndPoint(IPAddress.Any, 0);
    }

    void Update()
    {

    }

    void OnReceived(System.IAsyncResult result)
    {
        byte[] data = udpClient.EndReceive(result, ref remoteEP);
        string message = Encoding.UTF8.GetString(data);
        Debug.Log("Received: " + message);

        string[] coords = message.Split(' ');

        if(coords.Length == 3)
        {
            if (float.TryParse(coords[0], NumberStyles.Any, CultureInfo.InvariantCulture, out float x) &&
                float.TryParse(coords[1], NumberStyles.Any, CultureInfo.InvariantCulture, out float y) &&
                float.TryParse(coords[2], NumberStyles.Any, CultureInfo.InvariantCulture, out float z))
            {
                // this.transform.position = new Vector3(x, y, z);
                this.transform.localPosition = new Vector3(x, y, z);
            }
        }

        udpClient.BeginReceive(OnReceived, udpClient);
    }

    void OnApplicationQuit()
    {
        udpClient.Close();
    }
}