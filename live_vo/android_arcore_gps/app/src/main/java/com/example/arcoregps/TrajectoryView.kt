package com.example.arcoregps

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import kotlin.math.max
import kotlin.math.min

class TrajectoryView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val trajectoryPoints = mutableListOf<PointF>()
    private var isGpsMode = true
    
    // Paint objects for drawing
    private val gpsTrajectoryPaint = Paint().apply {
        color = Color.GREEN
        strokeWidth = 6f
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    
    private val voTrajectoryPaint = Paint().apply {
        color = Color.rgb(255, 102, 0) // Orange
        strokeWidth = 6f
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    
    private val gridPaint = Paint().apply {
        color = Color.rgb(40, 40, 40)
        strokeWidth = 1f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }
    
    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 24f
        isAntiAlias = true
    }
    
    // Trajectory bounds for auto-scaling
    private var minX = 0f
    private var maxX = 0f
    private var minZ = 0f
    private var maxZ = 0f
    private val margin = 50f

    fun addPoint(x: Float, y: Float, z: Float, gpsMode: Boolean) {
        // Only add point if it's significantly different from the last one (reduce noise)
        val newPoint = PointF(x, z)
        
        // Skip if too close to last point (less than 5cm movement)
        if (trajectoryPoints.isNotEmpty()) {
            val lastPoint = trajectoryPoints.last()
            val distance = kotlin.math.sqrt(
                ((newPoint.x - lastPoint.x) * (newPoint.x - lastPoint.x) + 
                 (newPoint.y - lastPoint.y) * (newPoint.y - lastPoint.y)).toDouble()
            ).toFloat()
            
            if (distance < 0.05f) { // 5cm threshold
                return // Skip this point
            }
        }
        
        trajectoryPoints.add(newPoint)
        isGpsMode = gpsMode
        
        // Update bounds with padding for better visualization
        if (trajectoryPoints.size == 1) {
            minX = x - 1.0f  // Add 1 meter padding
            maxX = x + 1.0f
            minZ = z - 1.0f
            maxZ = z + 1.0f
        } else {
            minX = min(minX, x - 0.5f)
            maxX = max(maxX, x + 0.5f)
            minZ = min(minZ, z - 0.5f)
            maxZ = max(maxZ, z + 0.5f)
            
            // Ensure minimum view size of 2x2 meters
            val centerX = (minX + maxX) / 2
            val centerZ = (minZ + maxZ) / 2
            val rangeX = maxX - minX
            val rangeZ = maxZ - minZ
            
            if (rangeX < 2.0f) {
                minX = centerX - 1.0f
                maxX = centerX + 1.0f
            }
            if (rangeZ < 2.0f) {
                minZ = centerZ - 1.0f
                maxZ = centerZ + 1.0f
            }
        }
        
        // Limit points to prevent memory issues
        if (trajectoryPoints.size > 500) { // Reduced from 1000
            trajectoryPoints.removeAt(0)
            // Recalculate bounds when removing old points
            recalculateBounds()
        }
        
        invalidate() // Trigger redraw
    }
    
    fun clearTrajectory() {
        trajectoryPoints.clear()
        minX = 0f
        maxX = 0f
        minZ = 0f
        maxZ = 0f
        invalidate()
    }
    
    fun setMode(gpsMode: Boolean) {
        isGpsMode = gpsMode
        invalidate()
    }
    
    private fun recalculateBounds() {
        if (trajectoryPoints.isEmpty()) return
        
        var newMinX = trajectoryPoints[0].x
        var newMaxX = trajectoryPoints[0].x
        var newMinZ = trajectoryPoints[0].y
        var newMaxZ = trajectoryPoints[0].y
        
        for (point in trajectoryPoints) {
            newMinX = min(newMinX, point.x)
            newMaxX = max(newMaxX, point.x)
            newMinZ = min(newMinZ, point.y)
            newMaxZ = max(newMaxZ, point.y)
        }
        
        // Add padding
        minX = newMinX - 0.5f
        maxX = newMaxX + 0.5f
        minZ = newMinZ - 0.5f
        maxZ = newMaxZ + 0.5f
        
        // Ensure minimum view size
        val centerX = (minX + maxX) / 2
        val centerZ = (minZ + maxZ) / 2
        val rangeX = maxX - minX
        val rangeZ = maxZ - minZ
        
        if (rangeX < 2.0f) {
            minX = centerX - 1.0f
            maxX = centerX + 1.0f
        }
        if (rangeZ < 2.0f) {
            minZ = centerZ - 1.0f
            maxZ = centerZ + 1.0f
        }
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        
        val viewWidth = width.toFloat()
        val viewHeight = height.toFloat()
        
        // Draw background
        canvas.drawColor(Color.BLACK)
        
        // Draw grid
        drawGrid(canvas, viewWidth, viewHeight)
        
        // Draw mode indicator
        val modeText = if (isGpsMode) "GPS Mode" else "Visual Odometry Mode"
        val modeColor = if (isGpsMode) Color.GREEN else Color.rgb(255, 102, 0)
        textPaint.color = modeColor
        canvas.drawText(modeText, 20f, 30f, textPaint)
        
        // Draw trajectory if we have points
        if (trajectoryPoints.size > 1) {
            drawTrajectory(canvas, viewWidth, viewHeight)
        }
        
        // Draw current position info and scale
        if (trajectoryPoints.isNotEmpty()) {
            val lastPoint = trajectoryPoints.last()
            val rangeX = maxX - minX
            val rangeZ = maxZ - minZ
            
            textPaint.color = Color.WHITE
            textPaint.textSize = 20f
            canvas.drawText("X: ${"%.2f".format(lastPoint.x)}m", 10f, viewHeight - 50f, textPaint)
            canvas.drawText("Z: ${"%.2f".format(lastPoint.y)}m", 120f, viewHeight - 50f, textPaint)
            canvas.drawText("Points: ${trajectoryPoints.size}", 230f, viewHeight - 50f, textPaint)
            
            // Draw scale info
            textPaint.textSize = 16f
            textPaint.color = Color.LTGRAY
            canvas.drawText("View: ${"%.1f".format(rangeX)}×${"%.1f".format(rangeZ)}m", 10f, viewHeight - 25f, textPaint)
            
            // Draw distance traveled
            if (trajectoryPoints.size > 1) {
                var totalDistance = 0f
                for (i in 1 until trajectoryPoints.size) {
                    val p1 = trajectoryPoints[i-1]
                    val p2 = trajectoryPoints[i]
                    val dx = p2.x - p1.x
                    val dy = p2.y - p1.y
                    totalDistance += kotlin.math.sqrt((dx*dx + dy*dy).toDouble()).toFloat()
                }
                canvas.drawText("Distance: ${"%.2f".format(totalDistance)}m", 180f, viewHeight - 25f, textPaint)
            }
        }
    }
    
    private fun drawGrid(canvas: Canvas, viewWidth: Float, viewHeight: Float) {
        val gridSize = 50f
        
        // Vertical lines
        var x = 0f
        while (x <= viewWidth) {
            canvas.drawLine(x, 0f, x, viewHeight, gridPaint)
            x += gridSize
        }
        
        // Horizontal lines
        var y = 0f
        while (y <= viewHeight) {
            canvas.drawLine(0f, y, viewWidth, y, gridPaint)
            y += gridSize
        }
    }
    
    private fun drawTrajectory(canvas: Canvas, viewWidth: Float, viewHeight: Float) {
        if (trajectoryPoints.isEmpty()) return
        
        // Calculate scaling to fit trajectory in view
        val dataWidth = maxX - minX
        val dataHeight = maxZ - minZ
        
        val scaleX = if (dataWidth > 0) (viewWidth - 2 * margin) / dataWidth else 1f
        val scaleY = if (dataHeight > 0) (viewHeight - 2 * margin) / dataHeight else 1f
        val scale = min(scaleX, scaleY)
        
        // Center the trajectory
        val centerX = viewWidth / 2
        val centerY = viewHeight / 2
        val dataCenterX = (minX + maxX) / 2
        val dataCenterZ = (minZ + maxZ) / 2
        
        // Choose paint based on current mode
        val paint = if (isGpsMode) gpsTrajectoryPaint else voTrajectoryPaint
        val linePaint = Paint().apply {
            color = paint.color
            style = Paint.Style.STROKE
            strokeWidth = 3f
            isAntiAlias = true
        }
        
        // Draw trajectory path - only draw every 3rd point to reduce clutter
        val path = Path()
        var pathStarted = false
        
        for (i in trajectoryPoints.indices step 3) { // Every 3rd point
            val point = trajectoryPoints[i]
            val screenX = centerX + (point.x - dataCenterX) * scale
            val screenY = centerY + (point.y - dataCenterZ) * scale
            
            // Ensure point is within view bounds
            if (screenX >= 0 && screenX <= viewWidth && screenY >= 0 && screenY <= viewHeight) {
                if (!pathStarted) {
                    path.moveTo(screenX, screenY)
                    pathStarted = true
                } else {
                    path.lineTo(screenX, screenY)
                }
                
                // Draw point - smaller and less frequent
                canvas.drawCircle(screenX, screenY, 3f, paint)
            }
        }
        
        // Draw path lines
        if (pathStarted) {
            canvas.drawPath(path, linePaint)
        }
        
        // Draw start point (larger, different color)
        if (trajectoryPoints.isNotEmpty()) {
            val startPoint = trajectoryPoints.first()
            val startX = centerX + (startPoint.x - dataCenterX) * scale
            val startY = centerY + (startPoint.y - dataCenterZ) * scale
            
            val startPaint = Paint().apply {
                color = Color.BLUE
                style = Paint.Style.FILL
                isAntiAlias = true
            }
            canvas.drawCircle(startX, startY, 8f, startPaint)
        }
        
        // Draw end point (larger, white)
        if (trajectoryPoints.isNotEmpty()) {
            val endPoint = trajectoryPoints.last()
            val endX = centerX + (endPoint.x - dataCenterX) * scale
            val endY = centerY + (endPoint.y - dataCenterZ) * scale
            
            val endPaint = Paint().apply {
                color = Color.WHITE
                style = Paint.Style.FILL
                isAntiAlias = true
            }
            canvas.drawCircle(endX, endY, 6f, endPaint)
        }
    }
    
    fun getTrajectoryData(): List<Triple<Float, Float, Float>> {
        return trajectoryPoints.mapIndexed { index, point ->
            Triple(point.x, 0f, point.y) // X, Y=0, Z
        }
    }
}