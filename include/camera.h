#ifndef CAMERA_IMPL
#define CAMERA_IMPL

enum class CameraMovement
{
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT,
    UP,
    DOWN
};

class Camera
{
private:
    // Spatial positioning and orientation vectors
    // These form the camera's local coordinate system in world space
    glm::vec3 position; // Camera's location in world coordinates
    glm::vec3 front;    // Forward direction (where camera is looking)
    glm::vec3 up;       // Camera's local up direction (for roll control)
    glm::vec3 right;    // Camera's local right direction (perpendicular to front and up)
    glm::vec3 worldUp;  // Global up vector reference (typically Y-axis)

    // Rotation representation using Euler angles
    // Provides intuitive control while managing gimbal lock and other rotation complexities
    float yaw;   // Horizontal rotation around the world up-axis (left-right looking)
    float pitch; // Vertical rotation around the camera's right axis (up-down looking)

    // User interaction and behavior parameters
    // These control how the camera responds to input and environmental factors
    float movementSpeed;    // Units per second for translation movement
    float mouseSensitivity; // Multiplier for mouse input to rotation angle conversion
    float zoom;             // Field of view control for perspective projection
    // Internal coordinate system maintenance
    // Ensures mathematical consistency when orientation changes occur
    void updateCameraVectors();

public:
    // Constructor with sensible defaults for common use cases
    // Provides flexibility while ensuring the camera starts in a predictable state
    Camera(
        glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f), // Start at world origin
        glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f),       // Y-axis as world up
        float yaw = -90.0f,                               // Look along negative Z-axis (OpenGL convention)
        float pitch = 0.0f                                // Level horizon
    );
    // Matrix generation for graphics pipeline integration
    // These methods bridge between the camera's spatial representation and GPU requirements
    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjectionMatrix(float aspectRatio, float nearPlane = 0.1f, float farPlane = 100.0f) const;

    // Input processing methods for different interaction modalities
    // Each method handles a specific type of user input with appropriate transformations
    void processKeyboard(CameraMovement direction, float deltaTime);                     // Keyboard-based translation
    void processMouseMovement(float xOffset, float yOffset, bool constrainPitch = true); // Mouse-based rotation
    void processMouseScroll(float yOffset);                                              // Scroll-based zoom control

    // Property access methods for external systems
    // Provide controlled access to internal state without exposing implementation details
    glm::vec3 getPosition() const;
    glm::vec3 getFront() const;
    float getZoom() const;
};

class ThirdPersonCamera : public Camera
{
private:
    // Target entity tracking and spatial relationship data
    // These properties define the relationship between camera and the character being followed
    glm::vec3 targetPosition; // Current world position of the target character
    glm::vec3 targetForward;  // Target's forward direction vector for contextual camera positioning

    // Camera behavior configuration parameters
    // These values control the aesthetic and functional characteristics of camera following
    float followDistance;   // Desired distance from target (affects intimacy and field of view)
    float followHeight;     // Height offset above target (provides better scene visibility)
    float followSmoothness; // Interpolation factor for smooth camera transitions (0 = instant, 1 = never)

    // Occlusion avoidance and collision management
    // These parameters control how the camera responds to environmental obstacles
    float minDistance;     // Minimum allowed distance from target (prevents camera from getting too close)
    float raycastDistance; // Maximum distance for occlusion detection rays

    // Internal computational state for smooth motion control
    // These variables manage the mathematical aspects of camera positioning and movement
    glm::vec3 desiredPosition;    // Target position the camera wants to reach (before collision adjustments)
    glm::vec3 smoothDampVelocity; // Velocity state for smooth damping interpolation algorithms

public:
    // Constructor with gameplay-tuned defaults
    // Default values chosen for common third-person game scenarios
    ThirdPersonCamera(
        float followDistance = 5.0f,   // Medium distance providing good character visibility and environment context
        float followHeight = 2.0f,     // Height above target for clear sightlines over low obstacles
        float followSmoothness = 0.1f, // Moderate smoothing for responsive but stable camera motion
        float minDistance = 1.0f       // Minimum distance to prevent uncomfortable close-ups
    );

    // Core functionality methods for camera behavior control
    void updatePosition(const glm::vec3 &targetPos, const glm::vec3 &targetFwd, float deltaTime);
    void handleOcclusion(const Scene &scene);
    void orbit(float horizontalAngle, float verticalAngle);

    // Runtime configuration methods for dynamic camera adjustment
    void setFollowDistance(float distance) { followDistance = distance; }
    void setFollowHeight(float height) { followHeight = height; }
    void setFollowSmoothness(float smoothness) { followSmoothness = smoothness; }
};

#endif