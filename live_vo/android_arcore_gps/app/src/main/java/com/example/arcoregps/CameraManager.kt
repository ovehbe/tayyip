package com.example.arcoregps

import android.content.Context
import android.graphics.ImageFormat
import android.hardware.camera2.*
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.util.Size
import android.view.Surface
import android.view.SurfaceHolder
import android.view.SurfaceView
import androidx.core.content.ContextCompat
// ARCore imports removed for now - will re-add when camera is working
// import com.google.ar.core.Session
// import com.google.ar.core.Config  
// import com.google.ar.core.Frame
// import com.google.ar.core.TrackingState
import java.util.Arrays

class CameraManager(
    private val context: Context,
    private val surfaceView: SurfaceView,
    private val onPoseUpdate: (FloatArray) -> Unit
) : SurfaceHolder.Callback {

    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var backgroundThread: HandlerThread? = null
    private var backgroundHandler: Handler? = null
    // private lateinit var arSession: Session  // Removed for now
    
    // Visual Odometry state
    private var isGpsMode = true
    private var lastPose: FloatArray? = null  // Simplified for now
    private val accumulatedTranslation = FloatArray(3)
    
    // Improved tracking simulation
    private var lastUpdateTime = 0L
    private val gpsPosition = FloatArray(3)        // GPS mode position
    private val voPosition = FloatArray(3)         // VO mode position  
    private val smoothedVoPosition = FloatArray(3) // Smoothed VO position

    init {
        surfaceView.holder.addCallback(this)
    }

    override fun surfaceCreated(holder: SurfaceHolder) {
        Log.d("CameraManager", "Surface created")
        startBackgroundThread()
        Log.d("CameraManager", "Background thread started, ready for camera")
    }

    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
        Log.d("CameraManager", "Surface changed: ${width}x${height}")
        setupCamera(width, height)
    }

    override fun surfaceDestroyed(holder: SurfaceHolder) {
        Log.d("CameraManager", "Surface destroyed")
        closeCamera()
        stopBackgroundThread()
    }

    private fun startBackgroundThread() {
        backgroundThread = HandlerThread("CameraBackground").also { it.start() }
        backgroundHandler = Handler(backgroundThread?.looper!!)
    }

    private fun stopBackgroundThread() {
        backgroundThread?.quitSafely()
        try {
            backgroundThread?.join()
            backgroundThread = null
            backgroundHandler = null
        } catch (e: InterruptedException) {
            Log.e("CameraManager", "Background thread interrupted", e)
        }
    }

    private fun setupCamera(width: Int, height: Int) {
        val cameraService = context.getSystemService(Context.CAMERA_SERVICE) as android.hardware.camera2.CameraManager
        try {
            val cameraId = cameraService.cameraIdList[0] // Use back camera
            val characteristics = cameraService.getCameraCharacteristics(cameraId)
            
            // Check permission
            if (ContextCompat.checkSelfPermission(context, android.Manifest.permission.CAMERA) 
                != android.content.pm.PackageManager.PERMISSION_GRANTED) {
                Log.e("CameraManager", "Camera permission not granted")
                return
            }

            cameraService.openCamera(cameraId, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    Log.d("CameraManager", "Camera opened")
                    cameraDevice = camera
                    createCameraPreviewSession()
                }

                override fun onDisconnected(camera: CameraDevice) {
                    Log.d("CameraManager", "Camera disconnected")
                    camera.close()
                    cameraDevice = null
                }

                override fun onError(camera: CameraDevice, error: Int) {
                    Log.e("CameraManager", "Camera error: $error")
                    camera.close()
                    cameraDevice = null
                }
            }, backgroundHandler)

        } catch (e: CameraAccessException) {
            Log.e("CameraManager", "Camera access exception", e)
        }
    }

    private fun createCameraPreviewSession() {
        try {
            val surface = surfaceView.holder.surface
            
            val captureRequestBuilder = cameraDevice?.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
            captureRequestBuilder?.addTarget(surface)

            cameraDevice?.createCaptureSession(
                Arrays.asList(surface),
                object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: CameraCaptureSession) {
                        Log.d("CameraManager", "Capture session configured")
                        captureSession = session
                        updatePreview()
                        startARCoreTracking()
                    }

                    override fun onConfigureFailed(session: CameraCaptureSession) {
                        Log.e("CameraManager", "Capture session configuration failed")
                    }
                }, backgroundHandler
            )

        } catch (e: CameraAccessException) {
            Log.e("CameraManager", "Camera access exception in preview", e)
        }
    }

    private fun updatePreview() {
        try {
            val surface = surfaceView.holder.surface
            val captureRequestBuilder = cameraDevice?.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
            captureRequestBuilder?.addTarget(surface)
            captureRequestBuilder?.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)

            captureSession?.setRepeatingRequest(
                captureRequestBuilder?.build()!!,
                null,
                backgroundHandler
            )
            Log.d("CameraManager", "Preview started")

        } catch (e: CameraAccessException) {
            Log.e("CameraManager", "Camera access exception in update preview", e)
        }
    }

    private fun startARCoreTracking() {
        // For now, just simulate pose tracking without ARCore to get camera working first
        Log.d("CameraManager", "Starting simulated tracking (ARCore disabled for now)")
        simulateTracking()
    }

    private fun simulateTracking() {
        // More realistic tracking simulation with different behavior for GPS vs VO
        backgroundHandler?.post {
            try {
                val currentTime = System.currentTimeMillis()
                val deltaTime = if (lastUpdateTime > 0) (currentTime - lastUpdateTime) / 1000f else 0f
                lastUpdateTime = currentTime
                
                if (isGpsMode) {
                    // GPS mode: Slower, more stable movement with occasional jumps
                    gpsPosition[0] += (Math.random() * 0.4 - 0.2).toFloat() * deltaTime
                    gpsPosition[1] += (Math.random() * 0.2 - 0.1).toFloat() * deltaTime
                    gpsPosition[2] += (Math.random() * 0.4 - 0.2).toFloat() * deltaTime
                    
                    // Occasional GPS "jump" to simulate GPS instability
                    if (Math.random() < 0.05) { // 5% chance
                        gpsPosition[0] += (Math.random() * 2.0 - 1.0).toFloat()
                        gpsPosition[2] += (Math.random() * 2.0 - 1.0).toFloat()
                    }
                    
                    onPoseUpdate(gpsPosition.clone())
                    Log.d("CameraManager", "GPS pose: X=${"%.3f".format(gpsPosition[0])}, Y=${"%.3f".format(gpsPosition[1])}, Z=${"%.3f".format(gpsPosition[2])}")
                    
                } else {
                    // VO mode: Smoother, more responsive movement (simulating camera tracking)
                    val moveSpeed = 0.8f
                    voPosition[0] += (Math.random() * 0.8 - 0.4).toFloat() * deltaTime * moveSpeed
                    voPosition[1] += (Math.random() * 0.4 - 0.2).toFloat() * deltaTime * moveSpeed
                    voPosition[2] += (Math.random() * 0.8 - 0.4).toFloat() * deltaTime * moveSpeed
                    
                    // Apply smoothing to VO
                    smoothedVoPosition[0] = smoothedVoPosition[0] * 0.8f + voPosition[0] * 0.2f
                    smoothedVoPosition[1] = smoothedVoPosition[1] * 0.8f + voPosition[1] * 0.2f
                    smoothedVoPosition[2] = smoothedVoPosition[2] * 0.8f + voPosition[2] * 0.2f
                    
                    onPoseUpdate(smoothedVoPosition.clone())
                    Log.d("CameraManager", "VO pose: X=${"%.3f".format(smoothedVoPosition[0])}, Y=${"%.3f".format(smoothedVoPosition[1])}, Z=${"%.3f".format(smoothedVoPosition[2])}")
                }
                
                // Schedule next update at 5 FPS to reduce noise
                backgroundHandler?.postDelayed({ simulateTracking() }, 200)
                
            } catch (e: Exception) {
                Log.e("CameraManager", "Error in simulated tracking", e)
                backgroundHandler?.postDelayed({ simulateTracking() }, 400)
            }
        }
    }

    private fun updateVisualOdometry(currentPose: FloatArray, translation: FloatArray) {
        val lastPose = this.lastPose
        
        if (lastPose != null) {
            // Calculate relative movement
            val deltaX = translation[0] - lastPose[0]
            val deltaY = translation[1] - lastPose[1]
            val deltaZ = translation[2] - lastPose[2]

            // Accumulate translation (simple integration)
            accumulatedTranslation[0] += deltaX
            accumulatedTranslation[1] += deltaY
            accumulatedTranslation[2] += deltaZ

            onPoseUpdate(accumulatedTranslation)
        }

        this.lastPose = currentPose.clone()
    }

    fun setGpsMode(enabled: Boolean) {
        isGpsMode = enabled
        if (!enabled) {
            // Initialize VO mode - start from current GPS position
            voPosition[0] = gpsPosition[0]
            voPosition[1] = gpsPosition[1] 
            voPosition[2] = gpsPosition[2]
            smoothedVoPosition[0] = gpsPosition[0]
            smoothedVoPosition[1] = gpsPosition[1]
            smoothedVoPosition[2] = gpsPosition[2]
        }
        Log.d("CameraManager", "GPS mode: $enabled")
    }

    fun resetTracking() {
        // Reset all position tracking
        gpsPosition[0] = 0f
        gpsPosition[1] = 0f
        gpsPosition[2] = 0f
        voPosition[0] = 0f
        voPosition[1] = 0f
        voPosition[2] = 0f
        smoothedVoPosition[0] = 0f
        smoothedVoPosition[1] = 0f
        smoothedVoPosition[2] = 0f
        accumulatedTranslation[0] = 0f
        accumulatedTranslation[1] = 0f
        accumulatedTranslation[2] = 0f
        lastPose = null
        lastUpdateTime = 0L
        Log.d("CameraManager", "All tracking reset")
    }

    private fun closeCamera() {
        try {
            captureSession?.close()
            captureSession = null
            cameraDevice?.close()
            cameraDevice = null
            Log.d("CameraManager", "Camera closed")
        } catch (e: Exception) {
            Log.e("CameraManager", "Error closing camera", e)
        }
    }

    fun pauseSession() {
        Log.d("CameraManager", "Session paused (ARCore disabled)")
        // TODO: Add ARCore pause when we re-enable it
    }

    fun resumeSession() {
        Log.d("CameraManager", "Session resumed (ARCore disabled)")
        // TODO: Add ARCore resume when we re-enable it
    }
}