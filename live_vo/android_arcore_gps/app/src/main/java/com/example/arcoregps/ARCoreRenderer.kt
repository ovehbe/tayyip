package com.example.arcoregps

import android.content.Context
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.opengl.Matrix
import android.view.Display
import android.view.WindowManager
import androidx.core.content.getSystemService
import com.google.ar.core.*
import com.google.ar.core.exceptions.CameraNotAvailableException
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

class ARCoreRenderer(
    private val context: Context,
    private val onPoseUpdate: (FloatArray) -> Unit
) : GLSurfaceView.Renderer {

    private var session: Session? = null
    private val backgroundRenderer = BackgroundRenderer()
    private val projectionMatrix = FloatArray(16)
    private val viewMatrix = FloatArray(16)
    private var displayRotationHelper: DisplayRotationHelper? = null

    // Visual Odometry state
    private var isGpsMode = true
    private var lastPose: Pose? = null
    private val accumulatedTranslation = FloatArray(3)

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES20.glClearColor(0.1f, 0.1f, 0.1f, 1.0f)

        try {
            // Initialize background renderer
            backgroundRenderer.createOnGlThread(context)
            
            // Create ARCore session
            session = Session(context)
            
            // Configure session for tracking
            val config = Config(session)
            config.planeFindingMode = Config.PlaneFindingMode.DISABLED
            config.lightEstimationMode = Config.LightEstimationMode.DISABLED
            session?.configure(config)

            displayRotationHelper = DisplayRotationHelper(context)

        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        displayRotationHelper?.onSurfaceChanged(width, height)
        GLES20.glViewport(0, 0, width, height)
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)

        val session = this.session ?: return
        val displayRotationHelper = this.displayRotationHelper ?: return

        try {
            session.setCameraTextureName(backgroundRenderer.textureId)

            // Update session
            val frame = session.update()
            val camera = frame.camera

            // Update display geometry
            displayRotationHelper.updateSessionIfNeeded(session)

            // Handle tracking state
            if (camera.trackingState == TrackingState.TRACKING) {
                // Get camera intrinsics and pose
                camera.getProjectionMatrix(projectionMatrix, 0, 0.1f, 100.0f)
                camera.getViewMatrix(viewMatrix, 0)

                val pose = camera.pose
                val translation = pose.translation

                if (isGpsMode) {
                    // In GPS mode, use ARCore pose directly
                    onPoseUpdate(translation)
                } else {
                    // In VO mode, accumulate relative motion
                    updateVisualOdometry(pose, translation)
                }

                // Render camera background
                backgroundRenderer.draw(frame)
            }

        } catch (e: CameraNotAvailableException) {
            e.printStackTrace()
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private fun updateVisualOdometry(currentPose: Pose, translation: FloatArray) {
        val lastPose = this.lastPose
        
        if (lastPose != null) {
            // Calculate relative movement
            val deltaX = translation[0] - lastPose.translation[0]
            val deltaY = translation[1] - lastPose.translation[1]
            val deltaZ = translation[2] - lastPose.translation[2]

            // Accumulate translation (simple integration)
            accumulatedTranslation[0] += deltaX
            accumulatedTranslation[1] += deltaY
            accumulatedTranslation[2] += deltaZ

            onPoseUpdate(accumulatedTranslation)
        }

        this.lastPose = currentPose
    }

    fun setGpsMode(enabled: Boolean) {
        isGpsMode = enabled
        if (!enabled && lastPose == null) {
            // Initialize VO mode - reset accumulated position
            accumulatedTranslation[0] = 0f
            accumulatedTranslation[1] = 0f
            accumulatedTranslation[2] = 0f
        }
    }

    fun resetTracking() {
        accumulatedTranslation[0] = 0f
        accumulatedTranslation[1] = 0f
        accumulatedTranslation[2] = 0f
        lastPose = null
    }

    fun pauseSession() {
        session?.pause()
    }

    fun resumeSession() {
        try {
            session?.resume()
        } catch (e: CameraNotAvailableException) {
            e.printStackTrace()
        }
    }
}

// Simple background renderer for camera feed
class BackgroundRenderer {
    var textureId = -1
        private set

    fun createOnGlThread(context: Context) {
        // Create texture for camera background
        val textures = IntArray(1)
        GLES20.glGenTextures(1, textures, 0)
        textureId = textures[0]
        
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureId)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
    }

    fun draw(frame: Frame) {
        try {
            // Draw camera background
            frame.transformCoordinates2d(
                Coordinates2d.OPENGL_NORMALIZED_DEVICE_COORDINATES,
                floatArrayOf(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f),
                Coordinates2d.TEXTURE_NORMALIZED,
                FloatArray(8)
            )
        } catch (e: Exception) {
            // Handle gracefully
        }
    }
}

// Helper class for display rotation
class DisplayRotationHelper(private val context: Context) {
    private var viewportChanged = false
    private var viewportWidth = 0
    private var viewportHeight = 0

    fun onSurfaceChanged(width: Int, height: Int) {
        viewportWidth = width
        viewportHeight = height
        viewportChanged = true
    }

    fun updateSessionIfNeeded(session: Session) {
        if (viewportChanged) {
            val windowManager = context.getSystemService<WindowManager>()
            val displayRotation = windowManager?.defaultDisplay?.rotation ?: 0
            session.setDisplayGeometry(displayRotation, viewportWidth, viewportHeight)
            viewportChanged = false
        }
    }
}