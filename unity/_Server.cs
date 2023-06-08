using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Net;
using System.Net.Sockets;
using System.Globalization;
using System.Text;

public class Server : MonoBehaviour
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
        string msg = Encoding.UTF8.GetString(data);
        Debug.Log(msg);
        
        udpClient.BeginReceive(OnReceived, udpClient);
    }
}