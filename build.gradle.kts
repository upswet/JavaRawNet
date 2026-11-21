plugins {
    `java-library`     
    `maven-publish`
    application
}

group = "io.github.upswet"  
version = "1.0.0"      

repositories {
    mavenCentral()
}

dependencies {
    // SLF4J
    implementation("org.slf4j:slf4j-api:2.0.20")
    implementation("org.slf4j:slf4j-simple:2.0.16")

    //lombok
    compileOnly("org.projectlombok:lombok:1.18.46")
    annotationProcessor("org.projectlombok:lombok:1.18.46")

    //=========================== TEST ===========================

    //lombok
    testCompileOnly("org.projectlombok:lombok:1.18.46")
    testAnnotationProcessor("org.projectlombok:lombok:1.18.46")

    // JUnit 5
    testImplementation("org.junit.jupiter:junit-jupiter-api:5.11.3")
    testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine:5.11.3")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(21))
    }
}

tasks.test {
    useJUnitPlatform()
}

publishing {
    publications {
        create<MavenPublication>("mavenJava") {
            from(components["java"])
        }
    }
}