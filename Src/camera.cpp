#include "main.h"
#include "camera.h"

glm::vec3 Camera::getPosition() const { return position; }
glm::vec3 Camera::getFront() const { return front; }
float Camera::getZoom() const { return zoom; }
Camera::Camera(
        glm::vec3 position,
        glm::vec3 up,
        float yaw,
        float pitch
    ){
        this->position = position;
        this->up = up;
        this->yaw = yaw;
        this->pitch = pitch;
    }
void Camera::processKeyboard(CameraMovement direction, float deltaTime)
{
    float velocity = movementSpeed * deltaTime;

    switch (direction)
    {
    case CameraMovement::FORWARD:
        position += front * velocity;
        break;
    case CameraMovement::BACKWARD:
        position -= front * velocity;
        break;
    case CameraMovement::LEFT:
        position -= right * velocity;
        break;
    case CameraMovement::RIGHT:
        position += right * velocity;
        break;
    case CameraMovement::UP:
        position += up * velocity;
        break;
    case CameraMovement::DOWN:
        position -= up * velocity;
        break;
    }
}
void Camera::processMouseMovement(float xOffset, float yOffset, bool constrainPitch)
{
    xOffset *= mouseSensitivity;
    yOffset *= mouseSensitivity;

    yaw += xOffset;
    pitch += yOffset;

    // Constrain pitch to avoid flipping
    if (constrainPitch)
    {
        pitch = std::clamp(pitch, -89.0f, 89.0f);
    }

    // Update camera vectors based on updated Euler angles
    updateCameraVectors();
}

void Camera::processMouseScroll(float yOffset)
{
    zoom -= (float)yOffset;
    if (zoom < 1.0f)
        zoom = 1.0f; // Clamp min FOV
    if (zoom > 60.0f)
        zoom = 60.0f; // Clamp max FOV
}
void Camera::updateCameraVectors()
{
    // Calculate the new front vector
    glm::vec3 newFront;
    newFront.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    newFront.y = sin(glm::radians(pitch));
    newFront.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    front = glm::normalize(newFront);

    // Recalculate the right and up vectors
    right = glm::normalize(glm::cross(front, worldUp));
    up = glm::normalize(glm::cross(right, front));
}
glm::mat4 Camera::getViewMatrix() const
{
    return glm::lookAt(position, position + front, up);
}
glm::mat4 Camera::getProjectionMatrix(float aspectRatio, float nearPlane, float farPlane) const
{
    return glm::perspective(glm::radians(zoom), aspectRatio, nearPlane, farPlane);
}

// void ThirdPersonCamera::updatePosition(
//     const glm::vec3& targetPos,
//     const glm::vec3& targetFwd,
//     float deltaTime
// ) {
//     // Update target properties
//     targetPosition = targetPos;
//     targetForward = glm::normalize(targetFwd);

//     // Calculate the desired camera position
//     // Position the camera behind and above the character
//     glm::vec3 offset = -targetForward * followDistance;
//     offset.y = followHeight;

//     desiredPosition = targetPosition + offset;

//     // Smooth camera movement using exponential smoothing
//     position = glm::mix(position, desiredPosition, 1.0f - pow(followSmoothness, deltaTime * 60.0f));

//     // Update the camera to look at the target
//     front = glm::normalize(targetPosition - position);

//     // Recalculate right and up vectors
//     right = glm::normalize(glm::cross(front, worldUp));
//     up = glm::normalize(glm::cross(right, front));
// }

// void ThirdPersonCamera::handleOcclusion(const Scene& scene) {
//     // Cast a ray from the target to the desired camera position
//     Ray ray;
//     ray.origin = targetPosition;
//     ray.direction = glm::normalize(desiredPosition - targetPosition);

//     // Check for intersections with scene objects
//     RaycastHit hit;
//     if (scene.raycast(ray, hit, glm::length(desiredPosition - targetPosition))) {
//         // If there's an intersection, move the camera to the hit point
//         // minus a small offset to avoid clipping
//         float offsetDistance = 0.2f;
//         position = hit.point - (ray.direction * offsetDistance);

//         // Ensure we don't get too close to the target
//         float currentDistance = glm::length(position - targetPosition);
//         if (currentDistance < minDistance) {
//             position = targetPosition + ray.direction * minDistance;
//         }

//         // Update the camera to look at the target
//         front = glm::normalize(targetPosition - position);
//         right = glm::normalize(glm::cross(front, worldUp));
//         up = glm::normalize(glm::cross(right, front));
//     }
// }

// void ThirdPersonCamera::orbit(float horizontalAngle, float verticalAngle) {
//     // Update yaw and pitch based on input
//     yaw += horizontalAngle;
//     pitch += verticalAngle;

//     // Constrain pitch to avoid flipping
//     pitch = std::clamp(pitch, -89.0f, 89.0f);

//     // Calculate the new camera position based on spherical coordinates
//     float radius = followDistance;
//     float yawRad = glm::radians(yaw);
//     float pitchRad = glm::radians(pitch);

//     // Convert spherical coordinates to Cartesian
//     glm::vec3 offset;
//     offset.x = radius * cos(yawRad) * cos(pitchRad);
//     offset.y = radius * sin(pitchRad);
//     offset.z = radius * sin(yawRad) * cos(pitchRad);

//     // Set the desired position
//     desiredPosition = targetPosition + offset;

//     // Update camera vectors
//     front = glm::normalize(targetPosition - desiredPosition);
//     right = glm::normalize(glm::cross(front, worldUp));
//     up = glm::normalize(glm::cross(right, front));
// }