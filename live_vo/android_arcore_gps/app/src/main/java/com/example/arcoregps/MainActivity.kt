package com.example.arcoregps

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.location.Location
import android.net.Uri
import android.view.SurfaceView
import android.os.Bundle
import android.os.Looper
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import com.google.android.gms.location.*
import com.google.ar.core.Session
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.*

class MainActivity : AppCompatActivity() {
    private lateinit var fusedLocationClient: FusedLocationProviderClient
    private var gpsOrigin: Location? = null
    private val poseLog = mutableListOf<Triple<Long, FloatArray, Location?>>()
    
    // UI components
    private lateinit var statusTextView: TextView
    private lateinit var poseTextView: TextView
    private lateinit var gpsTextView: TextView
    private lateinit var clearButton: Button
    private lateinit var stopGpsButton: Button
    private lateinit var restartGpsButton: Button
    
    // AR components
    private lateinit var surfaceView: SurfaceView
    private lateinit var cameraManager: CameraManager
    private lateinit var overlayStatusText: TextView
    private lateinit var overlayPoseText: TextView
    private lateinit var overlayGpsText: TextView
    
    // Trajectory visualization
    private lateinit var trajectoryView: TrajectoryView
    private lateinit var saveShareButton: Button
    
    // Mode tracking
    private var isGpsMode = true
    private var currentArPose = FloatArray(3)
    
    private val PERMISSIONS_REQUEST_CODE = 100
    private val REQUIRED_PERMISSIONS = arrayOf(
        Manifest.permission.CAMERA,
        Manifest.permission.ACCESS_FINE_LOCATION
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        // Initialize UI components
        initializeUI()
        
        // Check permissions
        if (!hasPermissions()) {
            requestPermissions()
        } else {
            initializeAR()
        }
    }
    
    private fun initializeUI() {
        statusTextView = findViewById(R.id.statusTextView)
        poseTextView = findViewById(R.id.poseTextView)
        gpsTextView = findViewById(R.id.gpsTextView)
        clearButton = findViewById(R.id.clearButton)
        stopGpsButton = findViewById(R.id.stopGpsButton)
        restartGpsButton = findViewById(R.id.restartGpsButton)
        
        // AR overlay components
        surfaceView = findViewById(R.id.surfaceView)
        overlayStatusText = findViewById(R.id.overlayStatusText)
        overlayPoseText = findViewById(R.id.overlayPoseText)
        overlayGpsText = findViewById(R.id.overlayGpsText)
        
        // Trajectory visualization
        trajectoryView = findViewById(R.id.trajectoryView)
        saveShareButton = findViewById(R.id.saveShareButton)
        
        // Button callbacks with GPS/VO mode switching
        clearButton.setOnClickListener {
            // Reset ARCore session and clear data
            try {
                cameraManager.resetTracking()
                poseLog.clear()
                gpsOrigin = null
                currentArPose = FloatArray(3)
                trajectoryView.clearTrajectory()
                updateOverlays()
                updateUI("Session cleared", "Pose reset", "GPS origin reset")
            } catch (e: Exception) {
                statusTextView.text = "Error clearing session: ${e.message}"
            }
        }
        
        stopGpsButton.setOnClickListener {
            // Switch to Visual Odometry mode
            isGpsMode = false
            fusedLocationClient.removeLocationUpdates(locCallback)
            cameraManager.setGpsMode(false)
            trajectoryView.setMode(false)
            updateOverlays()
            updateUI("Switched to Visual Odometry", null, "GPS stopped - using ARCore tracking")
        }
        
        restartGpsButton.setOnClickListener {
            // Switch back to GPS mode
            isGpsMode = true
            cameraManager.setGpsMode(true)
            trajectoryView.setMode(true)
            startLocationUpdates()
            updateOverlays()
            updateUI("Switched to GPS mode", null, "GPS restarted")
        }
        
        saveShareButton.setOnClickListener {
            saveAndShareTrajectory()
        }
    }
    
    private fun initializeAR() {
        try {
            // 1. Initialize GPS client
            fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)
            
            // 2. Setup Camera Manager with ARCore tracking
            cameraManager = CameraManager(this, surfaceView) { arPose ->
                currentArPose = arPose.clone()
                runOnUiThread { 
                    updateOverlays()
                    // Add point to trajectory visualization
                    trajectoryView.addPoint(arPose[0], arPose[1], arPose[2], isGpsMode)
                }
                
                // Log pose data
                val now = System.currentTimeMillis()
                poseLog.add(Triple(now, arPose, gpsOrigin))
            }
            
            // 3. Start location updates
            startLocationUpdates()
            
            updateUI("Camera and ARCore initialized", "Ready", "GPS starting...")
            updateOverlays()
            
        } catch (e: Exception) {
            statusTextView.text = "Failed to initialize: ${e.message}"
        }
    }
    
    private fun updateOverlays() {
        // Update camera overlay with current mode and position
        val modeText = if (isGpsMode) "GPS Mode" else "Visual Odometry Mode"
        val modeColor = if (isGpsMode) "#00FF00" else "#FF6600"
        
        overlayStatusText.text = modeText
        overlayStatusText.setTextColor(android.graphics.Color.parseColor(modeColor))
        
        // Update position display
        val poseText = "Position: X=${"%.2f".format(currentArPose[0])}, Y=${"%.2f".format(currentArPose[1])}, Z=${"%.2f".format(currentArPose[2])}"
        overlayPoseText.text = poseText
        
        // Update GPS status
        val gpsText = if (isGpsMode) {
            gpsOrigin?.let { 
                "GPS: Lat=${"%.6f".format(it.latitude)}, Lon=${"%.6f".format(it.longitude)}"
            } ?: "GPS: Waiting for signal..."
        } else {
            "GPS: Disabled (using visual tracking)"
        }
        overlayGpsText.text = gpsText
    }
    
    private fun startLocationUpdates() {
        if (ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.ACCESS_FINE_LOCATION
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            return
        }
        
        val locationRequest = LocationRequest.create().apply {
            interval = 1000
            fastestInterval = 500
            priority = LocationRequest.PRIORITY_HIGH_ACCURACY
        }
        
        fusedLocationClient.requestLocationUpdates(
            locationRequest,
            locCallback,
            Looper.getMainLooper()
        )
    }

    private val locCallback = object : LocationCallback() {
        override fun onLocationResult(result: LocationResult) {
            val loc = result.lastLocation
            if (loc != null) {
                if (gpsOrigin == null) {
                    gpsOrigin = loc
                    updateUI(null, null, "GPS origin set")
                }
                
                val gpsText = "GPS: Lat=${"%.6f".format(loc.latitude)}, Lon=${"%.6f".format(loc.longitude)}, Alt=${"%.2f".format(loc.altitude)}"
                updateUI(null, null, gpsText)
                updateOverlays()  // Update camera overlay with new GPS data
            }
        }
    }
    
    private fun updateUI(status: String?, pose: String?, gps: String?) {
        runOnUiThread {
            status?.let { statusTextView.text = it }
            pose?.let { poseTextView.text = it }
            gps?.let { gpsTextView.text = it }
        }
    }
    
    private fun hasPermissions(): Boolean {
        return REQUIRED_PERMISSIONS.all { permission ->
            ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED
        }
    }
    
    private fun requestPermissions() {
        ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, PERMISSIONS_REQUEST_CODE)
    }
    
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        
        if (requestCode == PERMISSIONS_REQUEST_CODE) {
            if (grantResults.all { it == PackageManager.PERMISSION_GRANTED }) {
                initializeAR()
            } else {
                statusTextView.text = "Permissions required for ARCore and GPS"
            }
        }
    }
    
    override fun onPause() {
        super.onPause()
        try {
            cameraManager.pauseSession()
        } catch (e: Exception) {
            // Handle if cameraManager not initialized yet
        }
        fusedLocationClient.removeLocationUpdates(locCallback)
    }
    
    override fun onResume() {
        super.onResume()
        try {
            cameraManager.resumeSession()
        } catch (e: Exception) {
            // Handle if cameraManager not initialized yet
        }
        if (hasPermissions() && isGpsMode) {
            startLocationUpdates()
        }
        updateOverlays()
    }

    override fun onDestroy() {
        super.onDestroy()
        // Write poseLog to CSV in external files dir as specified in the prompt
        savePoseLogToCSV()
    }
    
    private fun savePoseLogToCSV() {
        try {
            val externalFilesDir = getExternalFilesDir(null)
            val csvFile = File(externalFilesDir, "arcore_gps_trajectory.csv")
            
            FileWriter(csvFile).use { writer ->
                writer.append("timestamp,ar_x,ar_y,ar_z,gps_lat,gps_lon,gps_alt\n")
                
                poseLog.forEach { (timestamp, translation, gpsLoc) ->
                    writer.append("$timestamp,${translation[0]},${translation[1]},${translation[2]}")
                    if (gpsLoc != null) {
                        writer.append(",${gpsLoc.latitude},${gpsLoc.longitude},${gpsLoc.altitude}")
                    } else {
                        writer.append(",,,")
                    }
                    writer.append("\n")
                }
            }
            
            updateUI("CSV saved to: ${csvFile.absolutePath}", null, null)
        } catch (e: Exception) {
            updateUI("Error saving CSV: ${e.message}", null, null)
        }
    }
    
    private fun saveAndShareTrajectory() {
        try {
            // Create timestamp for filename
            val timestamp = SimpleDateFormat("yyyy-MM-dd_HH-mm-ss", Locale.getDefault()).format(Date())
            val filename = "trajectory_$timestamp.csv"
            
            // Save to external files directory
            val externalFilesDir = getExternalFilesDir(null)
            val csvFile = File(externalFilesDir, filename)
            
            FileWriter(csvFile).use { writer ->
                writer.append("frame,x,y,z,mode,timestamp\n")
                
                poseLog.forEachIndexed { index, (timestamp, translation, _) ->
                    val mode = if (isGpsMode) "GPS" else "VO"
                    writer.append("$index,${translation[0]},${translation[1]},${translation[2]},$mode,$timestamp\n")
                }
                
                // Also add trajectory view data
                val trajectoryData = trajectoryView.getTrajectoryData()
                trajectoryData.forEachIndexed { index, (x, y, z) ->
                    val mode = if (isGpsMode) "GPS" else "VO"
                    val currentTime = System.currentTimeMillis()
                    writer.append("traj_$index,$x,$y,$z,$mode,$currentTime\n")
                }
            }
            
            // Create share intent
            val contentUri = FileProvider.getUriForFile(
                this,
                "${applicationContext.packageName}.fileprovider",
                csvFile
            )
            
            val shareIntent = Intent().apply {
                action = Intent.ACTION_SEND
                type = "text/csv"
                putExtra(Intent.EXTRA_STREAM, contentUri)
                putExtra(Intent.EXTRA_SUBJECT, "Visual Odometry Trajectory - $timestamp")
                putExtra(Intent.EXTRA_TEXT, "Trajectory data captured with ${poseLog.size} points in ${if (isGpsMode) "GPS" else "Visual Odometry"} mode.")
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            
            val chooser = Intent.createChooser(shareIntent, "Share Trajectory Data")
            startActivity(chooser)
            
            Toast.makeText(this, "Trajectory saved! ${poseLog.size} points exported.", Toast.LENGTH_LONG).show()
            updateUI("Trajectory exported: $filename", null, null)
            
        } catch (e: Exception) {
            Toast.makeText(this, "Error saving trajectory: ${e.message}", Toast.LENGTH_LONG).show()
            updateUI("Export error: ${e.message}", null, null)
        }
    }
}